#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// STM32 side: GPIO, PWM, sensors, motors, and timing-sensitive operations.
// Linux/Qualcomm side: Python, AI inference, and high-level decisions.
//
// The challenge's 8x8 icons are centered on the UNO Q's onboard 8x13 matrix. No
// external pins or voltage assumptions are needed for the visual alert.

Arduino_LED_Matrix matrix;
bool alert_active = false;

constexpr uint8_t ICON_SMOKE[8] = {
  0x18, 0x3C, 0x7E, 0x7E, 0x7E, 0xFF, 0x18, 0x18
};
constexpr uint8_t ICON_GLASS[8] = {
  0x81, 0x52, 0x34, 0x18, 0x2C, 0x52, 0x81, 0xFF
};
constexpr uint8_t ICON_FALL[8] = {
  0x18, 0x18, 0x3C, 0x18, 0x38, 0x4C, 0x86, 0xFF
};

void draw_icon(const uint8_t rows[8]) {
  uint8_t frame[104] = {0};
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      // Two blank columns on the left and three on the right center the icon.
      frame[y * 13 + x + 2] = (rows[y] >> (7 - x)) & 0x01;
    }
  }
  matrix.draw(frame);
}

String health_check() {
  return String("ok");
}

void set_led(bool enabled) {
  // The built-in LED is active-low on the currently documented UNO Q example.
  digitalWrite(LED_BUILTIN, enabled ? LOW : HIGH);
}

String show_alert(String event) {
  if (event == "smoke_alarm") {
    draw_icon(ICON_SMOKE);
  } else if (event == "glass_break") {
    draw_icon(ICON_GLASS);
  } else if (event == "fall_thud") {
    draw_icon(ICON_FALL);
  } else {
    return String("unsupported event");
  }
  alert_active = true;
  return String("ok");
}

String show_spectrum(String columns) {
  // A compact, derived 13-band spectrum. This endpoint never receives
  // raw audio. Alert icons remain exclusive until clear_alert is called.
  if (alert_active) {
    return String("busy");
  }
  if (columns.length() != 13) {
    return String("invalid spectrum");
  }
  uint8_t frame[104] = {0};
  for (int x = 0; x < 13; ++x) {
    char value = columns.charAt(x);
    if (value < '0' || value > '8') {
      return String("invalid spectrum");
    }
    int height = value - '0';
    for (int y = 0; y < height; ++y) {
      frame[(7 - y) * 13 + x] = 1;
    }
  }
  matrix.draw(frame);
  return String("ok");
}

String clear_alert() {
  matrix.clear();
  alert_active = false;
  return String("ok");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  set_led(false);
  matrix.begin();
  matrix.setGrayscaleBits(1);
  matrix.clear();

  Bridge.begin();
  // provide_safe runs Arduino hardware APIs in the main loop context.
  Bridge.provide_safe("health_check", health_check);
  Bridge.provide_safe("set_led", set_led);
  Bridge.provide_safe("show_alert", show_alert);
  Bridge.provide_safe("show_spectrum", show_spectrum);
  Bridge.provide_safe("clear_alert", clear_alert);
}

void loop() {
  // Bridge callbacks are serviced by the runtime.
}
