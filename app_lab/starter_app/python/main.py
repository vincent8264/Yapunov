"""UNO Q Linux-side example. Run only inside Arduino App Lab on the board."""

import time

from arduino.app_utils import App, Bridge


def loop() -> None:
    # Replace this timer with the edge_ai pipeline's high-level Decision.
    # The STM32 owns GPIO/PWM/timing-sensitive work.
    Bridge.call("set_led", True)
    time.sleep(0.5)
    Bridge.call("set_led", False)
    time.sleep(0.5)


App.run(user_loop=loop)
