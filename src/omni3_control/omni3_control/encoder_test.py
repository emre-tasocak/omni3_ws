#!/usr/bin/env python3
"""
RoboClaw USB encoder okuma — her port ayri thread ile paralel okunur.
  /dev/ttyACM0 -> RoboClaw 1 (0x80): M1=On Sol, M2=On Sag
  /dev/ttyACM1 -> RoboClaw 2 (0x81): M2=Arka
"""

import time
import sys
import threading
from roboclaw import Roboclaw, RoboClawError

PORT_1 = "/dev/ttyACM0"
PORT_2 = "/dev/ttyACM1"
BAUD   = 38400
ADDR_1 = 0x80
ADDR_2 = 0x81
RATE   = 50   # Hz


def to_signed(val: int) -> int:
    return val if val < 2147483648 else val - 4294967296


class EncoderReader(threading.Thread):
    def __init__(self, port, baud, address, motors: list):
        super().__init__(daemon=True)
        self.address = address
        self.values  = {m: None for m in motors}
        self.errors  = {m: 0    for m in motors}
        self.motors  = motors
        self._rc     = Roboclaw(port, baud, timeout=0.1)
        print(f"[OK] {port}  addr=0x{address:02X}  baud={baud}")

    def run(self):
        while True:
            for m in self.motors:
                try:
                    fn = self._rc.ReadEncM1 if m == 1 else self._rc.ReadEncM2
                    val, ok = fn(self.address)
                    self.values[m] = to_signed(val) if ok else None
                    if not ok:
                        self.errors[m] += 1
                except Exception:
                    self.values[m] = None
                    self.errors[m] += 1
            time.sleep(1.0 / RATE)


def main():
    try:
        rc1 = EncoderReader(PORT_1, BAUD, ADDR_1, motors=[1, 2])
        rc2 = EncoderReader(PORT_2, BAUD, ADDR_2, motors=[2])
    except Exception as e:
        print(f"Port acilamadi: {e}")
        sys.exit(1)

    rc1.start()
    rc2.start()

    print(f"\nEncoder okuma basliyor ({RATE} Hz, Ctrl+C ile dur)\n")
    print(f"{'Sol (M1)':>14} | {'Sag (M2)':>14} | {'Arka (M2)':>14}   err(s/s/a)")
    print("-" * 65)

    try:
        while True:
            sol  = rc1.values[1]
            sag  = rc1.values[2]
            arka = rc2.values[2]

            sol_s  = f"{sol:>12}"  if sol  is not None else "      [HATA]"
            sag_s  = f"{sag:>12}"  if sag  is not None else "      [HATA]"
            arka_s = f"{arka:>12}" if arka is not None else "      [HATA]"

            e1, e2, e3 = rc1.errors[1], rc1.errors[2], rc2.errors[2]
            print(f"{sol_s} | {sag_s} | {arka_s}   err({e1}/{e2}/{e3})", end="\r")
            time.sleep(0.02)   # ekran 50 Hz yenileme

    except KeyboardInterrupt:
        print(f"\n\nSon degerler — Sol:{rc1.values[1]}  Sag:{rc1.values[2]}  Arka:{rc2.values[2]}")
        print(f"Hatalar    — Sol:{rc1.errors[1]}  Sag:{rc1.errors[2]}  Arka:{rc2.errors[2]}")


if __name__ == "__main__":
    main()
