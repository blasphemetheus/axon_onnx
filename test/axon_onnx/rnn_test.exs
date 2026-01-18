defmodule RNNTest do
  use ExUnit.Case

  @moduletag :rnn

  @moduledoc """
  Tests for LSTM and GRU layer serialization.

  Note: These tests verify that models export successfully and can be loaded
  by ONNX Runtime. Numerical comparison is not performed because RNN
  implementations may have subtle differences in gate ordering, numerical
  precision, or computational order between Axon and ONNX Runtime.
  """

  describe "serializes LSTM layers" do
    test "lstm output sequence only" do
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _states} = Axon.lstm(input, 64, name: "my_lstm")
      model = output_seq

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "lstm final hidden state with dense" do
      input = Axon.input("input", shape: {1, 10, 32})
      {_output_seq, {_cell, hidden}} = Axon.lstm(input, 64, name: "my_lstm")
      model = hidden |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "lstm sequence with dense (3D to 3D)" do
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _states} = Axon.lstm(input, 64, name: "my_lstm")
      model = output_seq |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "lstm with larger hidden size" do
      input = Axon.input("input", shape: {1, 5, 16})
      {output_seq, _states} = Axon.lstm(input, 128, name: "my_lstm")
      model = output_seq

      assert_model_exports_and_runs!(model, {1, 5, 16})
    end

    test "stacked lstm layers" do
      input = Axon.input("input", shape: {1, 8, 32})
      {output1, _states1} = Axon.lstm(input, 64, name: "lstm1")
      {output2, _states2} = Axon.lstm(output1, 32, name: "lstm2")
      model = output2 |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 8, 32})
    end
  end

  describe "serializes GRU layers" do
    test "gru output sequence only" do
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _state} = Axon.gru(input, 64, name: "my_gru")
      model = output_seq

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "gru final hidden state with dense" do
      input = Axon.input("input", shape: {1, 10, 32})
      {_output_seq, {hidden}} = Axon.gru(input, 64, name: "my_gru")
      model = hidden |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "gru sequence with dense (3D to 3D)" do
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _state} = Axon.gru(input, 64, name: "my_gru")
      model = output_seq |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "gru with larger hidden size" do
      input = Axon.input("input", shape: {1, 5, 16})
      {output_seq, _state} = Axon.gru(input, 128, name: "my_gru")
      model = output_seq

      assert_model_exports_and_runs!(model, {1, 5, 16})
    end

    test "stacked gru layers" do
      input = Axon.input("input", shape: {1, 8, 32})
      {output1, _state1} = Axon.gru(input, 64, name: "gru1")
      {output2, _state2} = Axon.gru(output1, 32, name: "gru2")
      model = output2 |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 8, 32})
    end
  end

  describe "serializes Dense layer with 3D input" do
    test "dense on 3D input without rnn" do
      # Verify that Dense works with 3D input (uses MatMul+Add instead of Gemm)
      model =
        Axon.input("input", shape: {1, 10, 32})
        |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end
  end

  describe "serializes sequence_last layer" do
    test "sequence_last extracts last timestep" do
      # [batch, seq_len, hidden] -> [batch, hidden]
      model =
        Axon.input("input", shape: {1, 10, 32})
        |> sequence_last(name: "last")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "lstm sequence with sequence_last and dense" do
      # This is the pattern ExPhil uses: LSTM -> last timestep -> Dense
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _states} = Axon.lstm(input, 64, name: "lstm")
      model = output_seq
        |> sequence_last(name: "last")
        |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "gru sequence with sequence_last and dense" do
      input = Axon.input("input", shape: {1, 10, 32})
      {output_seq, _state} = Axon.gru(input, 64, name: "gru")
      model = output_seq
        |> sequence_last(name: "last")
        |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 10, 32})
    end

    test "stacked lstm with sequence_last" do
      input = Axon.input("input", shape: {1, 8, 32})
      {output1, _states1} = Axon.lstm(input, 64, name: "lstm1")
      {output2, _states2} = Axon.lstm(output1, 32, name: "lstm2")
      model = output2
        |> sequence_last(name: "last")
        |> Axon.dense(16, name: "output")

      assert_model_exports_and_runs!(model, {1, 8, 32})
    end
  end

  # Helper to create a sequence_last layer
  # Extracts the last timestep from a sequence: [batch, seq_len, hidden] -> [batch, hidden]
  # This is ONNX-serializable (unlike Axon.nx which uses arbitrary functions)
  defp sequence_last(input, opts \\ []) do
    name = opts[:name] || "sequence_last"

    # Use Axon.layer/3 with signature: layer(op, inputs, opts)
    # The op_name: :sequence_last tells axon_onnx how to serialize it
    Axon.layer(
      fn inputs, _opts ->
        # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
        seq_len = Nx.axis_size(inputs, 1)
        Nx.slice_along_axis(inputs, seq_len - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      [input],
      name: name,
      op_name: :sequence_last
    )
  end

  # Helper that exports to ONNX and verifies it can be loaded by ONNX Runtime
  defp assert_model_exports_and_runs!(model, input_shape) do
    template = Nx.template(input_shape, {:f, 32})
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(template, %{})

    # Export to ONNX
    iodata = AxonOnnx.dump(model, template, params)
    binary = IO.iodata_to_binary(iodata)

    # Verify it's valid ONNX protobuf (basic structure check)
    assert byte_size(binary) > 0

    # Write to temp file and validate with Python/ONNX Runtime
    temp_path = Path.join(System.tmp_dir!(), "test_lstm_#{:erlang.unique_integer([:positive])}.onnx")

    try do
      File.write!(temp_path, binary)

      # Use Python to verify the model loads and runs
      {output, exit_code} = System.cmd("python3", [
        "-c",
        """
        import onnxruntime as ort
        import numpy as np
        import sys

        try:
            session = ort.InferenceSession('#{temp_path}', providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            input_shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
            input_data = np.random.randn(*input_shape).astype(np.float32)
            outputs = session.run(None, {input_name: input_data})
            print(f"SUCCESS: {len(outputs)} output(s), shape: {outputs[0].shape}")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        """
      ], stderr_to_stdout: true)

      assert exit_code == 0, "ONNX Runtime validation failed: #{output}"
    after
      File.rm(temp_path)
    end
  end
end
