"""Hardware-free end-to-end demo."""

import time

from edge_ai.decision import decide
from edge_ai.hardware.mock import MockHardware
from edge_ai.inference.dummy import DummyInferenceEngine
from edge_ai.inputs.simulated_sensor import SimulatedSensorInput
from edge_ai.pipeline import Pipeline
from edge_ai.preprocessing.sensor import normalize_sensor


def main() -> None:
    sensor = SimulatedSensorInput(seed=7)
    hardware = MockHardware(verbose=False)
    pipeline = Pipeline(
        input_source=sensor,
        preprocessor=normalize_sensor,
        inference=DummyInferenceEngine(),
        decision_function=decide,
        hardware=hardware,
    )

    print("Hardware-free Edge AI demo. Press Ctrl+C to stop.")
    try:
        while True:
            result, decision = pipeline.step()
            print(
                f"sensor={sensor.last_value:.2f} | "
                f"{result.label} {result.confidence:.2f} | action={decision.action}"
            )
            if decision.action == "alert":
                print("[MOCK] LED ON")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
