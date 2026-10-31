# Digital Smart Parking Counter — Project Context

## Objective
Build a ten-slot model on a hand-drawn A4 sheet (two one-way lanes, five angled slots per side) using **computer vision only** to detect toy vehicles. A phone camera streams to a laptop running Python/OpenCV from any reasonable angle. The laptop shows what it sees and calculates, and sends occupancy and gate commands via Wi-Fi to a NodeMCU ESP-12E (ESP8266), which drives an MG90S servo. A Flask dashboard and SQLite database show and record parking activity.

## Hardware
- NodeMCU ESP-12E (ESP8266), 3.3V logic, powered from laptop USB
- Phone camera (IP Webcam on Android, or DroidCam / Iriun / Camo) instead of the proposed Zebronics webcam
- One MG90S 180 degree positional servo for the entrance barrier on D5 (GPIO14)
- AA battery pack for the servo (check its rated voltage; use NiMH, a 1N4001 diode, or 3 AA) with a 470-1000 uF capacitor
- Common ground between servo battery negative and NodeMCU GND
- A4 sheet sketch and toy cars on a dark, non-white surface
- Later: 16x2 I2C LCD on D1/D2, red/green LEDs on D6/D7

**Do not use IR, LDR, ultrasonic or RFID sensors for vehicle detection.**

## Architecture
```text
Phone camera -> Laptop: Python/OpenCV -> Wi-Fi -> ESP8266 -> MG90S gate
                          |
                          +-> live vision dashboard
                          +-> Flask dashboard -> SQLite logs
```

## Detection and logic
- The sheet is found and warped to a top-down view; ten slot polygons (L1-L5, R1-R5) and an ENTRY zone are defined on the sheet.
- Each slot has binary state: `1` occupied, `0` empty.
- Occupied count = S1 + ... + S10 (arithmetic addition).
- Available = 10 - occupied count.
- FULL = AND of all ten slots.
- Gate permission = car_at_entrance AND NOT FULL.
- Detect passage and a clear barrier area before closing; include debouncing and timeouts.
- The OpenCV method compares each slot with an empty-sheet baseline (colour, brightness and new edges, with lighting normalisation and shadow tolerance); it is not object recognition.

## Website and logging
- Laptop runs Flask and SQLite.
- Dashboard shows available spaces and per-slot occupancy.
- Log slot occupancy sessions (occupied/empty timestamps), gate events and estimated durations.
- Do not claim to identify a particular vehicle without a separate identification method.

## Build order
1. Test NodeMCU and MG90S servo with AA battery power.
2. Run the vision dashboard with the phone stream and check sheet tracking.
3. Capture the empty baseline; tune slot polygons and thresholds.
4. Confirm stable detection from several camera angles.
5. Implement Wi-Fi messaging and ESP8266 gate control.
6. Add Flask live dashboard and SQLite database.
7. Add LCD/LEDs and test the complete model.

## Power and safety
- Never connect servo battery positive to NodeMCU 3V3 or VIN.
- Do not power MG90S from the NodeMCU 3V3 pin.
- Verify servo voltage rating before using a four-AA pack.
- Connect grounds together.
- Test the servo without a barrier arm first and ensure no obstruction before closing.

## Status
Vision pipeline is written and tested on simulated camera poses; purchases, wiring and real-camera tests have not been verified yet.
