#include <Arduino_RouterBridge.h>

// STM32 side: GPIO, PWM, sensors, motors, and timing-sensitive operations.
// Linux/Qualcomm side: Python, AI inference, and high-level decisions.
//
// No external pins or actuator endpoints are defined before the event. Add them only
// after verifying the supplied hardware, voltage levels, wiring, and Bridge version.

String health_check() {
  return String("ok");
}

void set_led(bool enabled) {
  // The built-in LED is active-low on the currently documented UNO Q example.
  digitalWrite(LED_BUILTIN, enabled ? LOW : HIGH);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  set_led(false);

  Bridge.begin();
  // provide_safe runs Arduino hardware APIs in the main loop context.
  Bridge.provide_safe("health_check", health_check);
  Bridge.provide_safe("set_led", set_led);
}

void loop() {
  // Bridge callbacks are serviced by the runtime.
}
