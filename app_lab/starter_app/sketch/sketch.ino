#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// STM32 side: GPIO, PWM, sensors, motors, and timing-sensitive operations.
// Linux/Qualcomm side: Python, AI inference, and high-level decisions.
//
// Alert layout on the UNO Q's onboard 8x13 matrix: columns 0-11 show the 8x8 icon
// (offset by two columns) on a background that blinks opposite to it; column 12 is a
// solid bar once the alert email is delivered. No external pins are used.
//
// Brightness levels assume the matrix accepts 3 grayscale bits (values 0-7). This is
// untested on the board; verify it against the installed Arduino_LED_Matrix library.

Arduino_LED_Matrix matrix;

constexpr int WIDTH = 13;
constexpr int HEIGHT = 8;
constexpr int ICON_AREA_WIDTH = 12;
constexpr int ICON_OFFSET = 2;
constexpr int STATUS_COLUMN = 12;
constexpr uint8_t GRAYSCALE_BITS = 3;
constexpr uint8_t LEVEL_OFF = 0;
constexpr uint8_t LEVEL_DIM = 1;
constexpr uint8_t LEVEL_BRIGHT = 7;
constexpr unsigned long BLINK_PHASE_MS = 500;

// Keep these rows in sync with ICONS in src/edge_ai/hardware/icons.py.
constexpr uint8_t ICON_SMOKE[8] = {
  0x49, 0x49, 0x92, 0x92, 0x49, 0x49, 0x92, 0x92
};
constexpr uint8_t ICON_GLASS[8] = {
  0x1E, 0x3C, 0x78, 0xFE, 0x1C, 0x38, 0x60, 0x80
};
constexpr uint8_t ICON_FALL[8] = {
  0x81, 0x42, 0x24, 0x18, 0x99, 0x7E, 0x18, 0x18
};
constexpr uint8_t ICON_HELP[8] = {
  0xC3, 0xC3, 0xC3, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3
};
constexpr uint8_t ICON_MICROPHONE_FAULT[8] = {
  0x19, 0x3D, 0x3E, 0x3C, 0xBD, 0x7E, 0x58, 0xBC
};

bool alert_active = false;
const uint8_t* active_icon = nullptr;
bool icon_bright = true;
bool email_delivered = false;
unsigned long last_toggle_ms = 0;

constexpr uint8_t STARTUP_CHECK[8] = {
  0x01, 0x03, 0x06, 0x8C, 0xD8, 0x70, 0x20, 0x00
};
constexpr uint8_t STARTUP_CROSS[8] = {
  0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81
};

void draw_alert() {
  uint8_t frame[WIDTH * HEIGHT] = {0};
  uint8_t icon_level = icon_bright ? LEVEL_BRIGHT : LEVEL_DIM;
  uint8_t background_level = icon_bright ? LEVEL_DIM : LEVEL_BRIGHT;
  uint8_t status_level = email_delivered ? LEVEL_BRIGHT : LEVEL_OFF;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < ICON_AREA_WIDTH; ++x) {
      int icon_x = x - ICON_OFFSET;
      bool lit = icon_x >= 0 && icon_x < 8 && ((active_icon[y] >> (7 - icon_x)) & 0x01);
      frame[y * WIDTH + x] = lit ? icon_level : background_level;
    }
    frame[y * WIDTH + STATUS_COLUMN] = status_level;
  }
  matrix.draw(frame);
}

String health_check() {
  return String("ok");
}

String show_startup_status(String status) {
  const uint8_t* icon = nullptr;
  uint8_t status_level = LEVEL_OFF;
  if (status == "checking") {
    // A small center dot means the board has responded and is checking audio.
    uint8_t frame[WIDTH * HEIGHT] = {0};
    frame[(HEIGHT / 2) * WIDTH + (WIDTH / 2)] = LEVEL_BRIGHT;
    matrix.draw(frame);
    return String("ok");
  }
  if (status == "ready" || status == "ready_offline") {
    icon = STARTUP_CHECK;
    // The final bar reports the advisory internet check: lit means reachable.
    status_level = status == "ready" ? LEVEL_BRIGHT : LEVEL_OFF;
  } else if (status == "failed") {
    icon = STARTUP_CROSS;
  } else {
    return String("unsupported startup status");
  }
  uint8_t frame[WIDTH * HEIGHT] = {0};
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < 8; ++x) {
      if ((icon[y] >> (7 - x)) & 0x01) {
        frame[y * WIDTH + x + ICON_OFFSET] = LEVEL_BRIGHT;
      }
    }
    frame[y * WIDTH + STATUS_COLUMN] = status_level;
  }
  matrix.draw(frame);
  return String("ok");
}

void set_led(bool enabled) {
  // The built-in LED is active-low on the currently documented UNO Q example.
  digitalWrite(LED_BUILTIN, enabled ? LOW : HIGH);
}

String show_alert(String event) {
  const uint8_t* icon = nullptr;
  if (event == "smoke_alarm") {
    icon = ICON_SMOKE;
  } else if (event == "glass_break") {
    icon = ICON_GLASS;
  } else if (event == "fall_thud") {
    icon = ICON_FALL;
  } else if (event == "help_call") {
    icon = ICON_HELP;
  } else if (event == "microphone_fault") {
    icon = ICON_MICROPHONE_FAULT;
  } else {
    return String("unsupported event");
  }
  // Repeated calls for the same alert keep the current blink phase and email status.
  if (!alert_active || icon != active_icon) {
    active_icon = icon;
    icon_bright = true;
    email_delivered = false;
    last_toggle_ms = millis();
    alert_active = true;
    draw_alert();
  }
  return String("ok");
}

String set_email_status(bool delivered) {
  // Ignored while no icon is shown; clear_alert resets the status for the next alert.
  if (alert_active) {
    email_delivered = delivered;
    draw_alert();
  }
  return String("ok");
}

String show_spectrum(String columns) {
  // A compact, derived 13-band spectrum. This endpoint never receives
  // raw audio. Alert icons remain exclusive until clear_alert is called.
  if (alert_active) {
    return String("busy");
  }
  if (columns.length() != WIDTH) {
    return String("invalid spectrum");
  }
  uint8_t frame[WIDTH * HEIGHT] = {0};
  for (int x = 0; x < WIDTH; ++x) {
    char value = columns.charAt(x);
    if (value < '0' || value > '8') {
      return String("invalid spectrum");
    }
    int height = value - '0';
    for (int y = 0; y < height; ++y) {
      frame[(HEIGHT - 1 - y) * WIDTH + x] = LEVEL_BRIGHT;
    }
  }
  matrix.draw(frame);
  return String("ok");
}

String clear_alert() {
  matrix.clear();
  alert_active = false;
  active_icon = nullptr;
  email_delivered = false;
  return String("ok");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  set_led(false);
  matrix.begin();
  matrix.setGrayscaleBits(GRAYSCALE_BITS);
  matrix.clear();

  Bridge.begin();
  // provide_safe runs Arduino hardware APIs in the main loop context.
  Bridge.provide_safe("health_check", health_check);
  Bridge.provide_safe("show_startup_status", show_startup_status);
  Bridge.provide_safe("set_led", set_led);
  Bridge.provide_safe("show_alert", show_alert);
  Bridge.provide_safe("set_email_status", set_email_status);
  Bridge.provide_safe("show_spectrum", show_spectrum);
  Bridge.provide_safe("clear_alert", clear_alert);
}

void loop() {
  // Bridge callbacks are serviced by the runtime; keep this loop non-blocking.
  if (alert_active && millis() - last_toggle_ms >= BLINK_PHASE_MS) {
    last_toggle_ms = millis();
    icon_bright = !icon_bright;
    draw_alert();
  }
}
