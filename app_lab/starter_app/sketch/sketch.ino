#include <Arduino_RouterBridge.h>

// STM32 side: GPIO, PWM, sensors, motors, and timing-sensitive operations.
// Linux/Qualcomm side: Python, AI inference, and high-level decisions.
//
// Pins are placeholders. Verify the UNO Q pinout, voltage levels, attached hardware,
// and the installed Bridge version before enabling PWM or a servo on a real board.

constexpr int PWM_PIN = 9;
constexpr int SERVO_PIN = 10;

void set_led(bool enabled) {
  // The built-in LED is active-low on the currently documented UNO Q example.
  digitalWrite(LED_BUILTIN, enabled ? LOW : HIGH);
}

void set_pwm(int channel, float value) {
  // Starter supports one placeholder channel; map channels once wiring is known.
  if (channel == 0) {
    value = constrain(value, 0.0f, 1.0f);
    analogWrite(PWM_PIN, static_cast<int>(value * 255.0f));
  }
}

void move_servo(int channel, float degrees) {
  // Pseudocode only: select and verify a UNO Q-compatible servo library first.
  // Example intent: servo[channel].write(constrain(degrees, 0.0f, 180.0f));
  (void)channel;
  (void)degrees;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(SERVO_PIN, OUTPUT);
  set_led(false);

  Bridge.begin();
  // provide_safe runs Arduino hardware APIs in the main loop context.
  Bridge.provide_safe("set_led", set_led);
  Bridge.provide_safe("set_pwm", set_pwm);
  Bridge.provide_safe("move_servo", move_servo);
}

void loop() {
  // Bridge callbacks are serviced by the runtime.
}
