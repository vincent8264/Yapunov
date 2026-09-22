# Models

Place hackathon ONNX models in this directory. Model binaries are git-ignored.

For every selected model, record:

- filename, source URL, version/hash, and license;
- input name, shape, layout, dtype, color order, resize/crop behavior, and normalization;
- label order;
- output names/shapes and whether outputs are logits, probabilities, embeddings, or
  another representation;
- the adapter and decision threshold used by this project;
- representative-device latency and any conversion or quantization steps.

Do not set `output_type = "probabilities"` merely to make configuration load. Confirm
the model actually includes the required sigmoid/softmax activation. For logits or
task-specific outputs, add and test a model-specific `output_adapter`.
