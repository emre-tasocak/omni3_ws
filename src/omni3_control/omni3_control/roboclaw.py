"""
RoboClaw Packet Serial surucusu — sadece pyserial bagimlidir.
Desteklenen komutlar: encoder okuma, motor hizi, reset.
"""

import struct
import serial
from threading import Lock


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        for bit in range(7, -1, -1):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ 0x1021
            else:
                crc = (crc << 1) & 0xFFFF
            if byte & (1 << bit):
                crc ^= 0x1021
    return crc


class RoboClawError(Exception):
    pass


class Roboclaw:
    def __init__(self, port: str, baud: int = 38400, timeout: float = 0.5):
        self._lock = Lock()
        self._ser = serial.Serial(port, baud, timeout=timeout)

    def close(self):
        self._ser.close()

    # ── düşük seviye ──────────────────────────────────────────────────────────

    def _read_cmd(self, address: int, cmd: int, fmt: str) -> tuple:
        """
        Okuma komutu: host sadece [address, cmd] gonderir, CRC eklemez.
        RoboClaw [data...][CRC16_high][CRC16_low] ile yanit verir.
        CRC, [address, cmd, data] uzerinden hesaplanir.
        """
        size = struct.calcsize(fmt)
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write(bytes([address, cmd]))
            raw = self._ser.read(size + 2)
        if len(raw) != size + 2:
            raise RoboClawError(f"Kisa yanit: beklenen {size + 2}, gelen {len(raw)}")
        payload, crc_bytes = raw[:size], raw[size:]
        crc_recv = struct.unpack(">H", crc_bytes)[0]
        crc_calc = _crc16(bytes([address, cmd]) + payload)
        if crc_recv != crc_calc:
            raise RoboClawError(f"CRC hatasi (beklenen 0x{crc_calc:04X}, gelen 0x{crc_recv:04X})")
        return struct.unpack(fmt, payload)

    def _write_cmd(self, address: int, cmd: int, fmt: str = "", *values) -> bool:
        """
        Yazma komutu: host [address, cmd, data, CRC16] gonderir.
        RoboClaw 0xFF ile onaylar.
        """
        data = struct.pack(fmt, *values) if fmt else b""
        packet = bytes([address, cmd]) + data
        crc = _crc16(packet)
        with self._lock:
            self._ser.write(packet + struct.pack(">H", crc))
            ack = self._ser.read(1)
        return len(ack) == 1 and ack[0] == 0xFF

    # ── encoder ───────────────────────────────────────────────────────────────

    def ReadEncM1(self, address: int) -> tuple[int, bool]:
        """(deger, gecerli_mi) döner. Status bit1=yon, bit2=overflow."""
        result = self._read_cmd(address, 0x10, ">IB")
        return result[0], True

    def ReadEncM2(self, address: int) -> tuple[int, bool]:
        result = self._read_cmd(address, 0x11, ">IB")
        return result[0], True

    # ── hız ──────────────────────────────────────────────────────────────────

    def ForwardM1(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 0, ">B", speed)

    def ForwardM2(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 4, ">B", speed)

    def BackwardM1(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 1, ">B", speed)

    def BackwardM2(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 5, ">B", speed)

    def SpeedM1(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 35, ">i", speed)

    def SpeedM2(self, address: int, speed: int) -> bool:
        return self._write_cmd(address, 36, ">i", speed)

    # ── PID ──────────────────────────────────────────────────────────────────

    def SetM1VelocityPID(self, address: int, p: float, i: float, d: float, qpps: int) -> bool:
        return self._write_cmd(address, 28, ">IIII",
                               int(d * 65536), int(p * 65536), int(i * 65536), int(qpps))

    def SetM2VelocityPID(self, address: int, p: float, i: float, d: float, qpps: int) -> bool:
        return self._write_cmd(address, 29, ">IIII",
                               int(d * 65536), int(p * 65536), int(i * 65536), int(qpps))

    # ── encoder sifirla ──────────────────────────────────────────────────────

    def ResetEncoders(self, address: int) -> bool:
        return self._write_cmd(address, 20)
