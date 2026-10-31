// Step 1: MG90S servo bench test on NodeMCU ESP-12E (ESP8266)
// Board in Arduino IDE: "NodeMCU 1.0 (ESP-12E Module)"
// Board package: ESP8266 by ESP8266 Community
//   Boards Manager URL: http://arduino.esp8266.com/stable/package_esp8266com_index.json
// Library: Servo.h is built into the ESP8266 core, nothing extra to install.
//
// Wiring:
//   Servo signal (orange/yellow) -> D5 (GPIO14)
//   Servo V+ (red)               -> AA pack positive (through a 1N4001 diode if the pack is 4 fresh alkalines)
//   Servo GND (brown)            -> AA pack negative AND NodeMCU GND (common ground)
//   470-1000 uF capacitor across servo V+ and GND, close to the servo
//   NEVER connect the AA pack positive to the 3V3 or VIN pin.
//
// Why D5: D1/D2 are the default I2C pins (kept free for the LCD later), and
// D3/D4/D8 are boot-strap pins that can stop the board from booting if loaded.
// Planned later pins: LEDs on D6 and D7.
//
// Serial Monitor: 115200 baud
//   o = open gate, c = close gate, m = middle (90), s = slow sweep, + / - = nudge 5 degrees

#include <Servo.h>

const int SERVO_PIN = D5;       // GPIO14
const int PULSE_MIN_US = 500;   // MG90S typical range 500-2400 us
const int PULSE_MAX_US = 2400;

// Tune these after the first test, with NO barrier arm attached yet
int ANGLE_CLOSED = 0;
int ANGLE_OPEN   = 90;

Servo gate;
int currentAngle = 90;

void moveTo(int target, int stepDelayMs = 15) {
  target = constrain(target, 0, 180);
  int dir = (target >= currentAngle) ? 1 : -1;
  while (currentAngle != target) {
    currentAngle += dir;
    gate.write(currentAngle);
    delay(stepDelayMs);   // slow move = lower current spikes, gentler on the barrier
  }
  Serial.printf("Angle now: %d\n", currentAngle);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  gate.attach(SERVO_PIN, PULSE_MIN_US, PULSE_MAX_US);
  gate.write(currentAngle);   // start at 90, a safe middle position
  Serial.println();
  Serial.println("Servo test ready. Commands: o c m s + -");
}

void loop() {
  if (!Serial.available()) return;
  char cmd = Serial.read();

  switch (cmd) {
    case 'o': moveTo(ANGLE_OPEN);   break;
    case 'c': moveTo(ANGLE_CLOSED); break;
    case 'm': moveTo(90);           break;
    case '+': moveTo(currentAngle + 5); break;
    case '-': moveTo(currentAngle - 5); break;
    case 's':
      moveTo(0, 20);
      moveTo(180, 20);
      moveTo(90, 20);
      break;
    default: break;
  }
  yield();   // keep the ESP8266 watchdog happy
}
