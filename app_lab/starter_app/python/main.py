"""UNO Q matrix smoke test. Run only inside Arduino App Lab on the board.

The production sound pipeline runs from the ``edge-ai`` package. This small App Lab
loop verifies Bridge and matrix behavior independently before enabling live audio.
"""

import time

from arduino.app_utils import App, Bridge

EVENTS = ("smoke_alarm", "glass_break", "fall_thud")
event_index = 0


def loop() -> None:
    global event_index
    event = EVENTS[event_index]
    response = Bridge.call("show_alert", event)
    if response != "ok":
        raise RuntimeError(f"matrix rejected {event!r}: {response!r}")
    print(f"matrix event: {event}")
    time.sleep(1.0)
    Bridge.call("clear_alert")
    time.sleep(0.25)
    event_index = (event_index + 1) % len(EVENTS)


App.run(user_loop=loop)
