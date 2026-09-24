# Build log 02: shared-audio help keyword scheduler

This log records the scheduler stage of the project. A local `help-kws.onnx`
prototype was trained later; its held-out results are in
[models/README.md](../../models/README.md#local-help-keyword-prototype).

We added the software boundary for a local spoken `help` detector while keeping the
existing environmental sound classifier. One microphone stream now feeds a rolling
one-second waveform. A small scheduler evaluates the keyword model every 200 ms and
YAMNet every third step, sequentially, then merges the results with `help_call`
priority. This avoids duplicate capture and avoids overlapping the two models' peak
CPU work.

The decision policy now tracks confirmation state per detector event. A negative
keyword window therefore cannot erase a pending smoke-alarm confirmation from the
less-frequent YAMNet schedule. The runner reports each detector's latency so the
schedule can be tuned from measurements on the physical UNO Q.

We also added two hardware-free tools: `inspect-audio` exposes YAMNet's raw AudioSet
labels, and `evaluate-keyword` calculates a confusion matrix plus precision, recall,
and accuracy from labeled WAV folders.

Current limitation: no trained `help` ONNX artifact has been selected. The adapter
enforces a documented raw-waveform/probability contract, and fake ONNX sessions cover
the integration path. Real accuracy and board latency remain unverified until a model,
recordings, and the event-supplied board are available.
