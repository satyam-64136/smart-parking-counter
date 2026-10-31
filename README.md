# Digital Smart Parking Counter

A four-slot miniature parking system that uses **computer vision only** to detect toy vehicles. A USB webcam feeds a laptop running Python and OpenCV, which sends occupancy data and gate commands over Wi-Fi to an ESP32 that drives an MG90S servo barrier. A Flask dashboard shows live status and SQLite records parking activity.

> **Status:** early prototyping. Hardware bench tests are the first milestone; nothing below is claimed as working until it has been tested.

## Architecture

```text
USB webcam -> Laptop: Python/OpenCV -> Wi-Fi -> ESP32 -> MG90S gate
                        |
                        +-> Flask dashboard -> SQLite logs
```

## Logic

- Each slot is binary: `1` occupied, `0` empty (S1..S4)
- Occupied count = S1 + S2 + S3 + S4
- Available = 4 - occupied count
- FULL = S1 AND S2 AND S3 AND S4
- Gate permission = vehicle_at_entrance AND NOT FULL

Detection starts with calibrated empty-scene image differencing. It sees visual change, not vehicle identity, and is sensitive to lighting.

## Hardware

- ESP32-WROOM-32 development board
- Zebronics Live Pro 1080p USB webcam
- MG90S servo (entrance barrier), powered from a separate battery pack with common ground
- About 40 x 40 cm cardboard model with four slots and toy cars
- Planned: 16x2 I2C LCD (level-shifted), red/green LEDs

### Power safety

- Never power the servo from the ESP32 3.3V pin
- Never connect the servo battery positive to the ESP32
- Connect battery negative to ESP32 GND
- Check the servo voltage rating (MG90S is about 4.8-6V) before using a four-AA pack

## Repository layout

```text
docs/        Project context and planning files
firmware/    ESP32 sketches (servo_test)
vision/      Python/OpenCV scripts (webcam_test.py)
```

## Getting started

### 1. Servo bench test (ESP32)

1. Install the **ESP32Servo** library in the Arduino IDE
2. Wire the servo signal to GPIO 18, battery power to the servo, and join grounds
3. Upload `firmware/servo_test/servo_test.ino` and use the Serial Monitor at 115200 baud (`o` open, `c` close, `m` middle, `s` sweep)

### 2. Webcam test (Python)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python vision/webcam_test.py
```

Press `s` to save empty-scene snapshots and `q` to quit.

## Roadmap

- [x] Project plan and architecture
- [ ] Servo bench test with battery power
- [ ] Webcam test with OpenCV
- [ ] Slot and entrance region calibration
- [ ] Multi-frame debouncing of occupancy
- [ ] Wi-Fi messaging and safe gate state machine
- [ ] Flask dashboard and SQLite logging
- [ ] LCD and LED integration, full demo

## Limitations

- Vision can fail from occlusion and lighting changes; always confirm the barrier is clear before closing
- The system logs slot occupancy sessions, not individual vehicle identities or true entry times
