defmodule AxonOnnx.Serialize do
  @moduledoc false

  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.NodeProto, as: Node
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorProto, as: Tensor
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  import AxonOnnx.Shared

  @onnx_ir_version 3
  @onnx_opset_version 14
  @producer_name "AxonOnnx"
  @producer_version "0.3.0"

  # Helper to extract shape from Axon.get_output_shape result
  # In Axon 0.8+, this returns a tensor template instead of a tuple
  defp extract_shape(%Nx.Tensor{} = tensor), do: Nx.shape(tensor)
  defp extract_shape(map) when is_map(map) do
    # Handle container outputs - extract shapes from each value
    Map.new(map, fn {k, v} -> {k, extract_shape(v)} end)
  end
  defp extract_shape(tuple) when is_tuple(tuple) do
    # Could be a shape tuple like {1, 10} or a tuple of outputs like {tensor, tensor}
    first = elem(tuple, 0)
    if is_struct(first, Nx.Tensor) or is_map(first) or (is_tuple(first) and tuple_size(first) > 0 and is_struct(elem(first, 0), Nx.Tensor)) do
      # Tuple of outputs - convert each
      tuple |> Tuple.to_list() |> Enum.map(&extract_shape/1)
    else
      # Plain shape tuple like {1, 10}
      tuple
    end
  end

  def __dump__(%Axon{} = axon, inputs, params, opts) do
    %Model{graph: %Graph{name: output_name}} =
      onnx_model = to_onnx_model(axon, inputs, params, opts)

    {Model.encode!(onnx_model), output_name}
  end

  defp to_onnx_model(axon, inputs, params, opts) do
    model_version = opts[:version] || 1
    doc_string = opts[:doc_string] || "An Axon Model"

    opset = %Opset{domain: "", version: @onnx_opset_version}

    graph = to_onnx_graph(axon, inputs, params)

    %Model{
      ir_version: @onnx_ir_version,
      producer_name: @producer_name,
      producer_version: @producer_version,
      domain: "",
      model_version: model_version,
      doc_string: doc_string,
      graph: graph,
      opset_import: [opset]
    }
  end

  defp to_onnx_graph(
         %Axon{output: id, nodes: nodes_map} = axon,
         templates,
         params_or_initializers
       ) do
    %Axon.Node{op: op, op_name: op_name, name: output_name_fn} = output_node = nodes_map[id]

    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(output_node, nodes_map, templates, [], [], [], %{}, %{})

    output_shape = Axon.get_output_shape(axon, templates) |> extract_shape()

    # Handle Axon.ModelState struct (Axon 0.8+ style)
    params_or_initializers =
      case params_or_initializers do
        %Axon.ModelState{data: data} -> data
        map when is_map(map) -> map
      end

    # Flatten params_or_initializers so it's no longer nested
    # Handle LSTM/GRU layers specially (they have nested gate weights)
    params_or_initializers =
      params_or_initializers
      |> Enum.reduce(%{}, fn {layer_name, params}, acc ->
        cond do
          # Detect LSTM/GRU by presence of input_kernel/hidden_kernel/bias as maps
          is_lstm_params?(params) ->
            flatten_lstm_params(acc, layer_name, params)

          true ->
            # Standard flattening for other layers
            params
            |> Enum.reduce(acc, fn {param_name, v}, acc ->
              Map.put(acc, layer_name <> "_" <> param_name, v)
            end)
        end
      end)

    # Building the initializers with Tensors will result in a bunch of expensive
    # copies, so we instead accumulate names and then use them to build initializers
    # later
    initializers = to_initializers(params_or_initializers, param_names)

    # Parameters need to be specified as graph inputs as well
    updated_inputs =
      param_names
      |> Enum.reduce(
        inputs,
        fn x, acc ->
          param_value = to_value_info(x, Nx.shape(params_or_initializers[x]))
          [param_value | acc]
        end
      )

    # Handle container (multi-output) vs single output models
    {graph_name, graph_outputs} =
      case op_name do
        :container ->
          # Container: multiple outputs from parent branches
          output_names = cache[id] || []
          parent_ids = output_node.parent

          outputs =
            Enum.zip(parent_ids, output_names)
            |> Enum.map(fn {pid, _name} ->
              parent_node = nodes_map[pid]
              # Get shape for this specific output
              parent_shape = get_parent_shape(output_shape, pid, parent_ids)
              {output_info, _, _} = to_value_info(parent_node, parent_shape, op_counts, cache)
              output_info
            end)

          # Use first output name as graph name
          {List.first(output_names) || "container", outputs}

        _ ->
          # Single output
          output_name =
            case cache do
              %{^id => name} -> name
              %{} -> output_name_fn.(op, op_counts)
            end

          {output_info, _, _} = to_value_info(output_node, output_shape, op_counts, cache)
          {output_name, [output_info]}
      end

    %Graph{
      node: Enum.reverse(nodes),
      name: graph_name,
      input: updated_inputs,
      output: graph_outputs,
      initializer: initializers
    }
  end

  # Helper to get shape for a specific parent in a container
  defp get_parent_shape(shape, _parent_id, _parent_ids) when is_tuple(shape), do: shape
  defp get_parent_shape(shapes, parent_id, parent_ids) when is_list(shapes) do
    # For list shapes (from tuple container), find by index
    idx = Enum.find_index(parent_ids, &(&1 == parent_id))
    Enum.at(shapes, idx) || List.first(shapes)
  end
  defp get_parent_shape(shapes, parent_id, parent_ids) when is_map(shapes) do
    # For map shapes (from map container), find the matching key by index
    idx = Enum.find_index(parent_ids, &(&1 == parent_id))
    keys = Map.keys(shapes)
    key = Enum.at(keys, idx)
    Map.get(shapes, key, shapes)
  end

  defp to_onnx(
         %Axon.Node{id: id, op: :constant, name: name, opts: [value: v]},
         _nodes_map,
         _templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    name = name.(:constant, op_counts)
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)
    cache = Map.put(cache, id, name)

    value_tensor = to_tensor_proto(v)
    value_attr = to_attr("value", :TENSOR, value_tensor)

    node = %Node{
      input: [],
      output: [name],
      name: name,
      op_type: "Constant",
      attribute: [value_attr]
    }

    {inputs, param_names, [node | nodes], op_counts, cache}
  end

  defp to_onnx(
         %Axon.Node{id: id, op: :input, name: name_fn} = axon,
         _nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # Check if already processed (prevents duplicate inputs in multi-branch models)
    case cache do
      %{^id => _name} ->
        # Already processed, skip
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        # TODO: Handle defaults
        name = name_fn.(:input, op_counts)

        shape =
          case templates do
            %Nx.Tensor{} = tensor ->
              Nx.shape(tensor)

            map ->
              Nx.shape(map[name])
          end

        {input_value, op_counts, cache} = to_value_info(axon, shape, op_counts, cache)
        {[input_value | inputs], param_names, nodes, op_counts, cache}
    end
  end

  ## Linear

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: :dense,
           name: name_fn,
           parent: [inp_id],
           parameters: params
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    # Early return if already processed by another branch (shared layers)
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(:dense, op_counts)
        op_counts = Map.update(op_counts, :dense, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        updated_param_names =
          Enum.map(params, fn %{name: p_name} ->
            name <> "_" <> p_name
          end)

        # Check input dimensionality - ONNX Gemm only supports 2D
        # For 3D input (e.g., from LSTM sequence), use MatMul + Add
        input_shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates) |> extract_shape()
        input_rank = tuple_size(input_shape)

        new_nodes = if input_rank == 3 do
          # 3D input: {batch, seq, features} -> {batch, seq, units}
          # Use MatMul (broadcasts over batch dims) + Add for bias
          kernel_name = name <> "_kernel"
          bias_name = name <> "_bias"
          matmul_output_name = name <> "_matmul"

          matmul_node = %Node{
            input: [inp_name, kernel_name],
            output: [matmul_output_name],
            name: name <> "_matmul_op",
            op_type: "MatMul"
          }

          add_node = %Node{
            input: [matmul_output_name, bias_name],
            output: [name],
            name: name,
            op_type: "Add"
          }

          [add_node, matmul_node]
        else
          # 2D input: use standard Gemm
          [%Node{
            input: [inp_name | updated_param_names],
            output: [name],
            name: name,
            op_type: "Gemm"
          }]
        end

        {inputs, updated_param_names ++ param_names, new_nodes ++ nodes, op_counts, cache}
    end
  end

  ## Convolution

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: :conv,
           name: name_fn,
           parent: [inp_id],
           parameters: params,
           opts: opts
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    # Early return if already processed by another branch (shared layers)
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(:conv, op_counts)
        op_counts = Map.update(op_counts, :conv, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        input_shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates) |> extract_shape()
        strides = opts[:strides] || 1
        strides = list_or_duplicate(:strides, strides, Nx.rank(input_shape) - 2)
        padding = opts[:padding]

        strides_attr = to_attr("strides", :INTS, strides)

        padding_attr =
          case padding do
            :valid ->
              to_attr("auto_pad", :STRING, "VALID")

            :same ->
              to_attr("auto_pad", :STRING, "SAME_UPPER")

            padding when is_list(padding) ->
              {pad_begins, pad_ends} = Enum.unzip(padding)
              to_attr("pads", :INTS, pad_begins ++ pad_ends)
          end

        # TODO: Dilations

        updated_param_names =
          Enum.map(params, fn %{name: p_name} ->
            name <> "_" <> p_name
          end)

        node = %Node{
          input: [inp_name | updated_param_names],
          output: [name],
          name: name,
          attribute: [strides_attr, padding_attr],
          op_type: "Conv"
        }

        {inputs, updated_param_names ++ param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Pooling

  @supported_pooling [:max_pool, :avg_pool, :lp_pool]

  defp to_onnx(
         %Axon.Node{id: id, op: pool, name: name_fn, parent: [inp_id], opts: opts},
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    # Early return if already processed by another branch (shared layers)
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(pool, op_counts)
        op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        input_shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates) |> extract_shape()

        kernel_size = tuple_or_duplicate(:kernel_size, opts[:kernel_size], Nx.rank(input_shape) - 2)
        strides = opts[:strides] || Tuple.to_list(kernel_size)
        strides = list_or_duplicate(:strides, strides, Nx.rank(input_shape) - 2)
        padding = opts[:padding]

        strides_attr = to_attr("strides", :INTS, strides)
        kernel_shape_attr = to_attr("kernel_shape", :INTS, Tuple.to_list(kernel_size))

        padding_attr =
          case padding do
            :valid ->
              to_attr("auto_pad", :STRING, "VALID")

            :same ->
              to_attr("auto_pad", :STRING, "SAME_UPPER")

            padding when is_list(padding) ->
              {pad_begins, pad_ends} = Enum.unzip(padding)
              to_attr("pads", :INTS, pad_begins ++ pad_ends)
          end

        # TODO: Dilations

        {op_type, extra_attrs} =
          case pool do
            :lp_pool ->
              p_attr = to_attr("p", :INT, opts[:norm])
              {"LpPool", [p_attr]}

            :max_pool ->
              {"MaxPool", []}

            :avg_pool ->
              count_include_pad_attr = to_attr("count_include_pad", :INT, 1)
              {"AveragePool", [count_include_pad_attr]}
          end

        node_inputs = [inp_name]

        node = %Node{
          input: node_inputs,
          output: [name],
          name: name,
          attribute: [padding_attr, strides_attr, kernel_shape_attr | extra_attrs],
          op_type: op_type
        }

        {inputs, param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Global Pooling

  @supported_global_pooling [:global_avg_pool, :global_lp_pool, :global_max_pool]

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: pool,
           name: name_fn,
           parent: [inp_id],
           opts: opts
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_global_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    # Early return if already processed by another branch (shared layers)
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(pool, op_counts)
        op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        keep_axes = opts[:keep_axes]

        {op_type, attrs} =
          case pool do
            :global_avg_pool ->
              {"GlobalAveragePool", []}

            :global_lp_pool ->
              {"GlobalLpPool", [to_attr("p", :INT, opts[:norm])]}

            :global_max_pool ->
              {"GlobalMaxPool", []}
          end

        node_inputs = [inp_name]

        nodes =
          if keep_axes do
            node = %Node{
              input: node_inputs,
              output: [name],
              name: name,
              attribute: attrs,
              op_type: op_type
            }

            [node | nodes]
          else
            pre_squeeze_name = name <> "_pre_squeeze"

            pre_squeeze_node = %Node{
              input: node_inputs,
              output: [pre_squeeze_name],
              name: pre_squeeze_name,
              attribute: attrs,
              op_type: op_type
            }

            constant_name = name <> "_squeeze_axes"
            shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates) |> extract_shape()
            axes = Enum.to_list(2..(tuple_size(shape) - 1)//1)
            axes_tensor = nx_to_tensor_proto(constant_name, Nx.tensor(axes))
            value_attr = to_attr("value", :TENSOR, axes_tensor)

            constant_node = %Node{
              output: [constant_name],
              name: constant_name,
              attribute: [value_attr],
              op_type: "Constant"
            }

            node = %Node{
              input: [pre_squeeze_name, constant_name],
              output: [name],
              name: name,
              op_type: "Squeeze"
            }

            [node, constant_node, pre_squeeze_node | nodes]
          end

        {inputs, param_names, nodes, op_counts, cache}
    end
  end

  ## Activations

  @supported_activations [
    {:celu, "Celu"},
    {:elu, "Elu"},
    {:exp, "Exp"},
    {:hard_sigmoid, "HardSigmoid"},
    {:leaky_relu, "LeakyRelu"},
    {:linear, "Identity"},
    {:relu, "Relu"},
    {:sigmoid, "Sigmoid"},
    {:selu, "Selu"},
    {:softmax, "Softmax"},
    {:softplus, "Softplus"},
    {:softsign, "Softsign"},
    {:tanh, "Tanh"}
  ]

  for {op, onnx_op} <- @supported_activations do
    defp to_onnx(
           %Axon.Node{id: id, op: unquote(op), name: name_fn, parent: [inp_id]},
           nodes_map,
           templates,
           inputs,
           param_names,
           nodes,
           op_counts,
           cache
         ) do
      {inputs, param_names, nodes, op_counts, cache} =
        to_onnx(
          nodes_map[inp_id],
          nodes_map,
          templates,
          inputs,
          param_names,
          nodes,
          op_counts,
          cache
        )

      input_name = cache[inp_id]

      # Early return if already processed by another branch (shared layers)
      case cache do
        %{^id => _} ->
          {inputs, param_names, nodes, op_counts, cache}

        %{} ->
          name = name_fn.(unquote(op), op_counts)
          op_counts = Map.update(op_counts, unquote(op), 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)

          node_inputs = [input_name]

          node = %Node{
            input: node_inputs,
            output: [name],
            name: name,
            op_type: unquote(onnx_op)
          }

          {inputs, param_names, [node | nodes], op_counts, cache}
      end
    end
  end

  ## Stochastic

  @supported_dropout_layers [:dropout, :spatial_droput, :feature_alpha_dropout, :alpha_dropout]

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: op,
           name: name_fn,
           parent: [inp_id]
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when op in @supported_dropout_layers do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    input_name = cache[inp_id]

    # Early return if already processed by another branch (shared layers)
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(op, op_counts)
        op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        # For now just forward with an identity
        node = %Node{
          input: [input_name],
          output: [name],
          name: name,
          op_type: "Identity"
        }

        # Just forward to the next layer
        {inputs, param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Container (multi-output models)

  defp to_onnx(
         %Axon.Node{id: id, op_name: :container, parent: parent_ids},
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # Container nodes group multiple outputs together.
    # We need to serialize all parent branches and collect their outputs.
    # The container itself doesn't create an ONNX node - it's just a grouping.

    {inputs, param_names, nodes, op_counts, cache} =
      Enum.reduce(parent_ids, {inputs, param_names, nodes, op_counts, cache}, fn parent_id, acc ->
        {inputs, param_names, nodes, op_counts, cache} = acc
        parent_node = nodes_map[parent_id]
        to_onnx(parent_node, nodes_map, templates, inputs, param_names, nodes, op_counts, cache)
      end)

    # Mark the container as processed by storing all parent output names
    output_names = Enum.map(parent_ids, fn pid -> cache[pid] end)
    cache = Map.put(cache, id, output_names)

    {inputs, param_names, nodes, op_counts, cache}
  end

  ## LSTM

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: :lstm,
           name: name_fn,
           parent: [input_id, _state_container_id, _index_id],
           parameters: _params,
           opts: _opts
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # Process input (skip state container and index - we'll use ONNX defaults)
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[input_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    input_name = cache[input_id]

    # Early return if already processed
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(:lstm, op_counts)
        op_counts = Map.update(op_counts, :lstm, 1, fn x -> x + 1 end)

        # LSTM in ONNX outputs: Y (all hidden states), Y_h (final hidden), Y_c (final cell)
        # We create output names for each
        output_seq_name = name <> "_output_sequence"
        output_h_name = name <> "_h"
        output_c_name = name <> "_c"

        # Cache the LSTM outputs as a tuple-like structure for :elem to extract
        cache = Map.put(cache, id, {output_seq_name, output_h_name, output_c_name})

        # Get hidden_size from output shape
        # LSTM returns [{seq_shape}, [{h_shape}, {c_shape}]]
        output_shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes_map}, templates) |> extract_shape()
        # Extract the sequence output shape: {batch, seq, hidden_size}
        seq_shape = case output_shape do
          [{batch, seq, hidden} | _rest] -> {batch, seq, hidden}
          {_batch, _seq, _hidden} = shape -> shape
          other -> raise "Unexpected LSTM output shape: #{inspect(other)}"
        end
        hidden_size = elem(seq_shape, 2)

        # Build ONNX LSTM attributes
        hidden_size_attr = to_attr("hidden_size", :INT, hidden_size)
        direction_attr = to_attr("direction", :STRING, "forward")
        # Note: layout=1 (batch-first) is not supported by ONNX Runtime
        # We'll use layout=0 (time-first) and add Transpose nodes

        # Parameter names for ONNX: W, R, B
        w_name = name <> "_W"
        r_name = name <> "_R"
        b_name = name <> "_B"

        updated_param_names = [w_name, r_name, b_name]

        # Create intermediate names
        transposed_input_name = name <> "_input_transposed"
        lstm_raw_output_name = name <> "_raw_output"
        squeezed_output_name = name <> "_squeezed"

        # 1. Transpose input: {batch, seq, features} -> {seq, batch, features}
        perm_attr_in = to_attr("perm", :INTS, [1, 0, 2])
        transpose_in_node = %Node{
          input: [input_name],
          output: [transposed_input_name],
          name: name <> "_transpose_in",
          attribute: [perm_attr_in],
          op_type: "Transpose"
        }

        # 2. Squeeze the num_directions dimension (axis 1): {seq, 1, batch, hidden} -> {seq, batch, hidden}
        squeeze_axes_name = name <> "_squeeze_axes"
        squeeze_axes_tensor = Nx.tensor([1], type: {:s, 64})
        squeeze_axes_value = to_attr("value", :TENSOR, to_tensor_proto(squeeze_axes_tensor))
        squeeze_axes_node = %Node{
          input: [],
          output: [squeeze_axes_name],
          name: squeeze_axes_name,
          attribute: [squeeze_axes_value],
          op_type: "Constant"
        }

        squeeze_node = %Node{
          input: [lstm_raw_output_name, squeeze_axes_name],
          output: [squeezed_output_name],
          name: name <> "_squeeze",
          op_type: "Squeeze"
        }

        # 4. Transpose output: {seq, batch, hidden} -> {batch, seq, hidden}
        perm_attr_out = to_attr("perm", :INTS, [1, 0, 2])
        transpose_out_node = %Node{
          input: [squeezed_output_name],
          output: [output_seq_name],
          name: name <> "_transpose_out",
          attribute: [perm_attr_out],
          op_type: "Transpose"
        }

        # 5. Squeeze Y_h: {num_directions=1, batch, hidden} -> {batch, hidden}
        # Reuse the squeeze axes constant
        lstm_raw_h_name = name <> "_raw_h"
        squeeze_h_axes_name = name <> "_squeeze_h_axes"
        squeeze_h_axes_tensor = Nx.tensor([0], type: {:s, 64})
        squeeze_h_axes_value = to_attr("value", :TENSOR, to_tensor_proto(squeeze_h_axes_tensor))
        squeeze_h_axes_node = %Node{
          input: [],
          output: [squeeze_h_axes_name],
          name: squeeze_h_axes_name,
          attribute: [squeeze_h_axes_value],
          op_type: "Constant"
        }

        squeeze_h_node = %Node{
          input: [lstm_raw_h_name, squeeze_h_axes_name],
          output: [output_h_name],
          name: name <> "_squeeze_h",
          op_type: "Squeeze"
        }

        # 6. Squeeze Y_c: {num_directions=1, batch, hidden} -> {batch, hidden}
        lstm_raw_c_name = name <> "_raw_c"
        squeeze_c_axes_name = name <> "_squeeze_c_axes"
        squeeze_c_axes_tensor = Nx.tensor([0], type: {:s, 64})
        squeeze_c_axes_value = to_attr("value", :TENSOR, to_tensor_proto(squeeze_c_axes_tensor))
        squeeze_c_axes_node = %Node{
          input: [],
          output: [squeeze_c_axes_name],
          name: squeeze_c_axes_name,
          attribute: [squeeze_c_axes_value],
          op_type: "Constant"
        }

        squeeze_c_node = %Node{
          input: [lstm_raw_c_name, squeeze_c_axes_name],
          output: [output_c_name],
          name: name <> "_squeeze_c",
          op_type: "Squeeze"
        }

        # Update LSTM node to output to raw names (before squeeze)
        lstm_node = %Node{
          input: [transposed_input_name, w_name, r_name, b_name],
          output: [lstm_raw_output_name, lstm_raw_h_name, lstm_raw_c_name],
          name: name,
          attribute: [hidden_size_attr, direction_attr],
          op_type: "LSTM"
        }

        new_nodes = [
          transpose_out_node, squeeze_node, squeeze_axes_node,
          squeeze_h_node, squeeze_h_axes_node,
          squeeze_c_node, squeeze_c_axes_node,
          lstm_node, transpose_in_node
        ]

        {inputs, updated_param_names ++ param_names, new_nodes ++ nodes, op_counts, cache}
    end
  end

  ## Recurrent State (skip - ONNX uses zeros by default)

  defp to_onnx(
         %Axon.Node{id: id, op_name: :recurrent_state, parent: parent_ids},
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # Process parents first
    {inputs, param_names, nodes, op_counts, cache} =
      Enum.reduce(parent_ids, {inputs, param_names, nodes, op_counts, cache}, fn parent_id, acc ->
        {inputs, param_names, nodes, op_counts, cache} = acc
        parent_node = nodes_map[parent_id]
        to_onnx(parent_node, nodes_map, templates, inputs, param_names, nodes, op_counts, cache)
      end)

    # Recurrent state is handled by ONNX LSTM defaults (zeros)
    # Just mark as processed with a placeholder
    cache = Map.put(cache, id, "__recurrent_state_#{id}__")

    {inputs, param_names, nodes, op_counts, cache}
  end

  ## Constant

  defp to_onnx(
         %Axon.Node{id: id, op: :constant, name: name_fn, opts: opts},
         _nodes_map,
         _templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        name = name_fn.(:constant, op_counts)
        op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)
        cache = Map.put(cache, id, name)

        # Get the constant value
        value = opts[:value]

        # Create a Constant node
        tensor_proto = to_tensor_proto(value)
        value_attr = to_attr("value", :TENSOR, tensor_proto)

        node = %Node{
          input: [],
          output: [name],
          name: name,
          attribute: [value_attr],
          op_type: "Constant"
        }

        {inputs, param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Elem (extract element from tuple output)

  defp to_onnx(
         %Axon.Node{id: id, op_name: :elem, parent: [parent_id], opts: opts},
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # Process parent first
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[parent_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    case cache do
      %{^id => _} ->
        {inputs, param_names, nodes, op_counts, cache}

      %{} ->
        # The parent should have cached a tuple of output names
        parent_outputs = cache[parent_id]

        # For LSTM, we need to figure out which output this elem extracts
        # The index might be in opts, or we need to infer from output shape
        # LSTM outputs: {output_seq (3D), h (2D), c (2D)}
        this_shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes_map}, templates) |> extract_shape()

        # Determine which LSTM output this is based on shape
        output_name = case parent_outputs do
          {seq, h, _c} ->
            # Infer from shape: output_seq is 3D, h and c are 2D
            case this_shape do
              {_, _, _} -> seq  # 3D -> output sequence
              {_, _} ->
                # 2D could be h or c - check opts or default to h
                # In most cases, users want the hidden state (h)
                opts[:index] || h
            end
          name when is_binary(name) -> name
          _ -> "elem_#{id}"
        end

        # Cache this node with the extracted output name
        cache = Map.put(cache, id, output_name)

        # No new ONNX node needed - we're just aliasing an existing output
        {inputs, param_names, nodes, op_counts, cache}
    end
  end

  defp to_attr(name, type, value) do
    case type do
      :INT ->
        %Attribute{name: name, type: :INT, i: value}

      :INTS ->
        %Attribute{name: name, type: :INTS, ints: value}

      :STRING ->
        %Attribute{name: name, type: :STRING, s: value}

      :TENSOR ->
        %Attribute{name: name, type: :TENSOR, t: value}
    end
  end

  defp to_initializers(params_or_initializers, param_names) do
    param_names
    |> Enum.map(fn param ->
      nx_to_tensor_proto(param, params_or_initializers[param])
    end)
  end

  defp to_value_info(%Axon.Node{id: id, op: op, name: name_fn}, shape, op_counts, cache) do
    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    {%Value{name: name, type: input_type}, op_counts, cache}
  end

  defp to_value_info(param_name, shape) do
    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    %Value{name: param_name, type: input_type}
  end

  defp to_placeholder(shape) do
    %Placeholder{shape: to_tensor_shape_proto(shape), elem_type: 1}
  end

  defp to_tensor_proto(tensor) do
    dims = Tuple.to_list(Nx.shape(tensor))
    type = nx_type_to_onnx_type(Nx.type(tensor))
    data = Nx.to_binary(tensor)

    %Tensor{dims: dims, data_type: type, raw_data: data}
  end

  defp to_tensor_shape_proto(shape) do
    # Handle both tuple shapes and tensor templates (Axon 0.8+ compat)
    shape_tuple = case shape do
      %Nx.Tensor{} -> Nx.shape(shape)
      tuple when is_tuple(tuple) -> tuple
    end

    dims =
      shape_tuple
      |> Tuple.to_list()
      |> Enum.map(fn
        nil ->
          %Dimension{value: {:dim_param, 1}}

        value ->
          %Dimension{value: {:dim_value, value}}
      end)

    %Shape{dim: dims}
  end

  defp nx_to_tensor_proto(param_name, tensor) do
    dims = Nx.shape(tensor) |> Tuple.to_list()
    # TODO: fix
    data_type =
      case Nx.type(tensor) do
        {:f, 32} ->
          1

        {:s, 64} ->
          7
      end

    raw_data = Nx.to_binary(tensor)
    %Onnx.TensorProto{name: param_name, dims: dims, data_type: data_type, raw_data: raw_data}
  end

  defp tuple_or_duplicate(key, tuple_or_integer, rank) do
    cond do
      is_tuple(tuple_or_integer) ->
        if tuple_size(tuple_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element tuple, " <>
                  "got: #{inspect(tuple_or_integer)}"
        end

        tuple_or_integer

      is_integer(tuple_or_integer) ->
        Tuple.duplicate(tuple_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a tuple, " <>
                "got: #{inspect(tuple_or_integer)}"
    end
  end

  defp list_or_duplicate(key, list_or_integer, rank) do
    cond do
      is_list(list_or_integer) ->
        if length(list_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element list, " <>
                  "got: #{inspect(list_or_integer)}"
        end

        list_or_integer

      is_integer(list_or_integer) ->
        List.duplicate(list_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a list, " <>
                "got: #{inspect(list_or_integer)}"
    end
  end

  ## LSTM/GRU Weight Helpers

  # Detect if params structure is from an LSTM/GRU layer
  defp is_lstm_params?(params) when is_map(params) do
    has_input_kernel = Map.has_key?(params, "input_kernel")
    has_hidden_kernel = Map.has_key?(params, "hidden_kernel")
    has_bias = Map.has_key?(params, "bias")

    # Check if input_kernel is a map with gate keys (not a tensor)
    input_kernel = params["input_kernel"]
    input_kernel_is_map = is_map(input_kernel) and not is_struct(input_kernel, Nx.Tensor)

    has_input_kernel and has_hidden_kernel and has_bias and input_kernel_is_map
  end

  defp is_lstm_params?(_), do: false

  # Flatten LSTM params into ONNX format: W, R, B
  # Axon gate order: i (input), f (forget), g (cell), o (output)
  # ONNX gate order: i (input), o (output), f (forget), c (cell)
  defp flatten_lstm_params(acc, layer_name, params) do
    input_kernel = params["input_kernel"]
    hidden_kernel = params["hidden_kernel"]
    bias = params["bias"]

    # Extract individual gate weights from Axon's nested structure
    # Input kernels: {input_size, hidden_size} each
    wii = input_kernel["wii"]
    wif = input_kernel["wif"]
    wig = input_kernel["wig"]
    wio = input_kernel["wio"]

    # Hidden kernels: {hidden_size, hidden_size} each
    whi = hidden_kernel["whi"]
    whf = hidden_kernel["whf"]
    whg = hidden_kernel["whg"]
    who = hidden_kernel["who"]

    # Biases: {hidden_size} each
    bi = bias["bi"]
    bf = bias["bf"]
    bg = bias["bg"]
    bo = bias["bo"]

    # Build ONNX W tensor: [num_directions, 4*hidden_size, input_size]
    # Reorder gates: Axon (i,f,g,o) -> ONNX (i,o,f,c)
    # Transpose from {input_size, hidden_size} to {hidden_size, input_size}
    w = build_lstm_weight_matrix([wii, wio, wif, wig])

    # Build ONNX R tensor: [num_directions, 4*hidden_size, hidden_size]
    r = build_lstm_weight_matrix([whi, who, whf, whg])

    # Build ONNX B tensor: [num_directions, 8*hidden_size]
    # ONNX has input bias and hidden bias concatenated, but Axon only has one bias
    # We'll use Axon's bias for input bias and zeros for hidden bias
    b = build_lstm_bias([bi, bo, bf, bg])

    acc
    |> Map.put(layer_name <> "_W", w)
    |> Map.put(layer_name <> "_R", r)
    |> Map.put(layer_name <> "_B", b)
  end

  # Build W or R matrix: concatenate gate weights and add num_directions dim
  # Input: list of 4 tensors in ONNX gate order (i, o, f, c)
  # Each tensor: {in_size, hidden_size} for W, {hidden_size, hidden_size} for R
  # Output: {1, 4*hidden_size, in_size}
  defp build_lstm_weight_matrix([wi, wo, wf, wc]) do
    # Transpose each from {in, hidden} to {hidden, in}
    wi_t = Nx.transpose(wi)
    wo_t = Nx.transpose(wo)
    wf_t = Nx.transpose(wf)
    wc_t = Nx.transpose(wc)

    # Concatenate along axis 0: {4*hidden_size, in_size}
    concatenated = Nx.concatenate([wi_t, wo_t, wf_t, wc_t], axis: 0)

    # Add num_directions dimension: {1, 4*hidden_size, in_size}
    Nx.reshape(concatenated, {1, elem(Nx.shape(concatenated), 0), elem(Nx.shape(concatenated), 1)})
  end

  # Build B tensor: [num_directions, 8*hidden_size]
  # ONNX format: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
  # Axon only has one set of biases, so we use zeros for hidden biases (Rb_*)
  defp build_lstm_bias([bi, bo, bf, bc]) do
    hidden_size = elem(Nx.shape(bi), 0)

    # Concatenate input biases
    input_bias = Nx.concatenate([bi, bo, bf, bc], axis: 0)

    # Create zero hidden biases
    hidden_bias = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(bi)), {4 * hidden_size})

    # Concatenate: {8*hidden_size}
    full_bias = Nx.concatenate([input_bias, hidden_bias], axis: 0)

    # Add num_directions dimension: {1, 8*hidden_size}
    Nx.reshape(full_bias, {1, elem(Nx.shape(full_bias), 0)})
  end
end
