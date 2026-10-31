// Step 1: MG90S servo bench test on ESP32-WROOM-32
// Library: "ESP32Servo" by Kevin Harrington (Arduino Library Manager)
//
// Wiring:
//   Servo orange/yellow (signal) -> GPIO 18
//   Servo red (V+)               -> AA pack positive (through the diode, see notes)
//   Servo brown (GND)            -> AA pack negative AND ESP32 GND (common ground)
//   NEVER connect the AA pack positive to the ESP32 3.3V or 5V/VIN pin.
//
// Serial Monitor: 115200 baud, line ending "No line ending" or "Newline"
//   o = open gate, c = close gate, m = middle (90), s = slow sweep, + / - = nudge 5 degrees

#include <ESP32Servo.h>

const int SERVO_PIN = 18;
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

  ESP32PWM::allocateTimer(0);
  gate.setPeriodHertz(50);                       // standard servo frame rate
  gate.attach(SERVO_PIN, PULSE_MIN_US, PULSE_MAX_US);

  gate.write(currentAngle);                      // start at 90, a safe middle position
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
}
