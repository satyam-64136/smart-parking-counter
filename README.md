# Digital Smart Parking Counter

A ten-slot miniature parking system that uses **computer vision only** to detect toy vehicles. A phone camera streams to a laptop running Python and OpenCV, which shows exactly what it sees and calculates, and sends occupancy and gate commands over Wi-Fi to a NodeMCU ESP-12E (ESP8266) that drives an MG90S servo barrier. A Flask dashboard and SQLite logging are planned next.

> **Status:** prototyping. The vision pipeline is tested on simulated camera angles (`vision/selftest.py`); real-camera and hardware tests are still to do.

## How the vision works

1. Find the white A4 sheet in the camera frame and refine its four corners
2. Warp it to a flat top-down view, so the camera angle stops mattering
3. Normalise lighting using the lane pixels as a brightness reference
4. Compare with a baseline photo of the empty sheet (colour, brightness, new edges; shadows tolerated)
5. Per slot: changed pixels / slot pixels = occupancy ratio
6. Debounce with hysteresis so a slot flips only after several consistent frames
7. Digital logic: occupied = sum of slots, available = 10 - occupied, FULL = AND of all slots, gate opens if ENTRY AND NOT FULL

Detection sees visual change, not vehicle identity.

## Layout

Two one-way lanes around a central divider (left lane up, right lane down), five angled slots per side: `L1-L5` and `R1-R5`, plus an `ENTRY` zone at the top of the down lane. Slot polygons are stored relative to the sheet, so they work from any angle. Edit them with the `E` key in the dashboard (saved to `slots.json`).

## Hardware

- NodeMCU ESP-12E (ESP8266)
- Phone camera (Android "IP Webcam", or DroidCam / Iriun / Camo)
- MG90S servo on D5 (GPIO14), powered from a separate AA pack with common ground
- A4 sheet sketch and toy cars on a dark, non-white surface
- Planned: 16x2 I2C LCD on D1/D2, red/green LEDs on D6/D7

### Power safety

- Never power the servo from the NodeMCU 3V3 pin and never connect the battery positive to 3V3 or VIN
- Connect battery negative to NodeMCU GND
- MG90S is about 4.8-6V: use 4 NiMH, 3 AA, or put a 1N4001 diode in series with 4 fresh alkalines
- Add a 470-1000 uF capacitor across the servo supply

## Repository layout

```text
docs/        Project context files
firmware/    NodeMCU (ESP8266) sketches: servo_test
vision/      parking_vision.py (dashboard), webcam_test.py, selftest.py
```

## Getting started

### 1. Servo bench test (NodeMCU)

1. Arduino IDE: add the ESP8266 board package, select **NodeMCU 1.0 (ESP-12E Module)**
2. Wire the signal to D5, battery power to the servo, join grounds
3. Upload `firmware/servo_test/servo_test.ino`, open Serial Monitor at 115200 baud (`o` open, `c` close, `m` middle, `s` sweep)

### 2. Vision dashboard

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python vision/selftest.py                                    # no camera needed
python vision/parking_vision.py --source http://PHONE_IP:8080/video
python vision/parking_vision.py --source 1                   # DroidCam / Iriun / webcam index
python vision/parking_vision.py --source photo.jpg           # test on a saved photo
```

Keys: `B` baseline (sheet empty), `L` lock/unlock sheet tracking, `F` flip 180, `R` rotate 90, `E` edit slots, `S` screenshot, `Q` quit.

Workflow: show the whole sheet with all slots empty, check the yellow outline and that the top-down view is upright (press `F` if upside down), press `B`, then place toy cars.

## Roadmap

- [x] Project plan and architecture
- [x] Vision pipeline with live processing dashboard (simulation-tested)
- [ ] Servo bench test with battery power
- [ ] Real phone-camera test and threshold tuning
- [ ] Wi-Fi messaging and safe gate state machine
- [ ] Flask dashboard and SQLite logging
- [ ] LCD and LED integration, full demo

## Limitations

- Tall toy cars shift in the top-down view (parallax): mount the phone high and keep slot polygons snug
- Vision can fail from occlusion and heavy shadows; confirm the barrier is clear before closing
- The system logs slot occupancy sessions, not vehicle identities or true entry times
