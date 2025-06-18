# AutoCAD prism structure generator (stair and triangle modes)

import math
import time
import os
import sys
import warnings
from dataclasses import dataclass

from pyautocad import Autocad, APoint
import comtypes

sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore", UserWarning)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def retry_autocad_call(func, retries: int = 3, wait_time: int = 5):
    """Call ``func`` and retry if AutoCAD is temporarily busy."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:  # pragma: no cover - direct COM calls
            print(f"AutoCAD call failed, retry {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise


def send_command_with_retry(acad, command: str, retries: int = 5, delay: int = 2):
    """Send a raw command string to AutoCAD with retry logic."""
    for attempt in range(retries):
        try:
            acad.ActiveDocument.SendCommand(command)
            break
        except comtypes.COMError as e:  # pragma: no cover - direct COM calls
            print(f"Failed to send command to AutoCAD (attempt {attempt + 1}): {e}")
            time.sleep(delay)
    else:
        raise RuntimeError(f"Failed to execute command after retries: {command}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OutputPaths:
    """Paths used when exporting AutoCAD files."""

    folder: str
    sat_name: str = "prism_sat_file-sim.SAT"
    dwg_name: str = "Drawing.dwg"
    center_y_name: str = "center_y.txt"
    center_x_name: str = "center_x.txt"

    @property
    def sat_path(self) -> str:
        return os.path.join(self.folder, self.sat_name)

    @property
    def dwg_path(self) -> str:
        return os.path.join(self.folder, self.dwg_name)

    @property
    def center_y_path(self) -> str:
        return os.path.join(self.folder, self.center_y_name)

    @property
    def center_x_path(self) -> str:
        return os.path.join(self.folder, self.center_x_name)


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class PrismBuilder:
    """Create prism structures in AutoCAD."""

    def __init__(self, scale: float = 1.0, pixel_size: int = 22, sleep_time: float = 0.2):
        self.scale = scale
        self.pixel_size = pixel_size
        self.sleep_time = sleep_time

        # Launch AutoCAD and create a new document
        self.acad = retry_autocad_call(lambda: Autocad(create_if_not_exists=True))
        retry_autocad_call(lambda: self.acad.app.Documents.Add())
        while self.acad.ActiveDocument is None:
            time.sleep(1)
        time.sleep(self.sleep_time)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _triangle_points(self, side_a: float, side_b: float, angle_B: float):
        angle_B2_rad = math.radians(90 - angle_B)
        B = (0, side_a)
        Cx = side_b * math.cos(angle_B2_rad)
        Cy = side_a - side_b * math.sin(angle_B2_rad)
        C = (Cx, Cy)
        A = (0, 0)
        side_c = math.sqrt(side_a ** 2 + side_b ** 2 - 2 * side_a * side_b * math.cos(math.radians(angle_B)))
        Ix = (side_b * A[0] + side_c * B[0] + side_a * Cx) / (side_a + side_b + side_c)
        Iy = (side_b * A[1] + side_c * B[1] + side_a * Cy) / (side_a + side_b + side_c)
        return A, B, C, Cx, Cy, Ix, Iy

    def _draw_triangle(self, A, B, C):
        for p1, p2 in zip([A, B, C], [B, C, A]):
            self.acad.model.AddLine(
                APoint(p1[0] * self.scale, p1[1] * self.scale),
                APoint(p2[0] * self.scale, p2[1] * self.scale),
            )

    def _draw_stair(self, equ_ac, equ_bc, bottom: float, top: float):
        current_x1 = current_x2 = 0
        current_y1, current_y2 = bottom, top
        pos1, pos2 = [], []
        while True:
            equ_y1 = equ_ac(current_x1)
            equ_y2 = equ_bc(current_x2)
            real_y1 = math.ceil(equ_y1 / self.pixel_size) * self.pixel_size
            real_y2 = math.floor(equ_y2 / self.pixel_size) * self.pixel_size

            if real_y2 != current_y2 and current_y1 != real_y2:
                current_x2 -= self.pixel_size
                pos2.extend([(current_x2, current_y2), (current_x2, real_y2)])
            elif real_y2 == real_y1:
                break
            else:
                pos2.extend([(current_x2 - self.pixel_size, real_y2), (current_x2, real_y2)])

            if real_y1 != current_y1 and current_y1 != real_y2:
                current_x1 -= self.pixel_size
                pos1.extend([(current_x1, current_y1), (current_x1, real_y1)])
            elif real_y1 == real_y2:
                break
            else:
                pos1.extend([(current_x1 - self.pixel_size, real_y1), (current_x1, real_y1)])

            if current_x1 <= current_x2:
                current_x1 += self.pixel_size
            else:
                current_x1 -= self.pixel_size
            current_x2 += self.pixel_size
            current_y1 = real_y1
            current_y2 = real_y2

        for seq in [pos1, pos2]:
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    self.acad.model.AddLine(
                        APoint(seq[i][0] * self.scale, seq[i][1] * self.scale),
                        APoint(seq[i + 1][0] * self.scale, seq[i + 1][1] * self.scale),
                    )

        self.acad.model.AddLine(
            APoint(pos1[0][0] * self.scale, bottom * self.scale),
            APoint(pos1[0][0] * self.scale, top * self.scale),
        )
        self.acad.model.AddLine(
            APoint(pos1[-1][0] * self.scale, pos1[-1][1] * self.scale),
            APoint(pos2[-1][0] * self.scale, pos2[-1][1] * self.scale),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, sid_ang, mode: str, paths: OutputPaths) -> None:
        side_a = round(sid_ang[0], 2)
        side_b = round(sid_ang[1], 2)
        angle_B = sid_ang[2]
        A, B, C, Cx, Cy, Ix, Iy = self._triangle_points(side_a, side_b, angle_B)

        slope_ac = Cy / Cx if Cx != 0 else 0
        slope_bc = (Cy - B[1]) / (Cx - B[0]) if (Cx - B[0]) != 0 else 0
        intercept_bc = B[1] - slope_bc * B[0]
        equ_ac = lambda x: slope_ac * x
        equ_bc = lambda x: slope_bc * x + intercept_bc
        top = (math.floor(equ_bc(0) / self.pixel_size)) * self.pixel_size
        bottom = (math.ceil(equ_ac(0) / self.pixel_size)) * self.pixel_size

        if mode == "triangle":
            top = equ_bc(0)
            bottom = equ_ac(0)
            self._draw_triangle(A, B, C)
        elif mode == "stair":
            self._draw_stair(equ_ac, equ_bc, bottom, top)
        else:
            raise ValueError("mode must be 'stair' or 'triangle'")

        send_command_with_retry(self.acad, "SELECT\nALL\n\n_JOIN\n\n")
        send_command_with_retry(self.acad, "ZOOM\nE\n\n")
        send_command_with_retry(self.acad, f"-BOUNDARY\n{Ix},{Iy}\n\n")
        send_command_with_retry(self.acad, "_EXTRUDE\nL\n\n1\n")
        time.sleep(self.sleep_time)
        send_command_with_retry(self.acad, "UNION\nALL\n\n")

        rows, columns = 30, 1
        row_spacing = side_a * self.scale * (rows - 1)
        column_spacing = 1
        send_command_with_retry(
            self.acad,
            f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
        )

        time.sleep(self.sleep_time)
        send_command_with_retry(self.acad, "Explode\nALL\n\n")
        send_command_with_retry(self.acad, "UNION\nALL\n\n")

        light_source_length = 0.5
        actual_array_top = top + (rows - 1) * (top - bottom)
        array_center_y = (actual_array_top + bottom) / 2
        center_y = round(array_center_y * self.scale, 1)
        center_x = round(Cx * self.scale + 1, 1)
        with open(paths.center_y_path, "w") as f:
            f.write(str(center_y))
        with open(paths.center_x_path, "w") as f:
            f.write(str(center_x))

        start_point = APoint(100, array_center_y * self.scale, 0)
        send_command_with_retry(
            self.acad,
            f"_BOX\n{start_point.x},{start_point.y},{start_point.z}\n{start_point.x + light_source_length + 0.6},{start_point.y + light_source_length + 0.6},{start_point.z + light_source_length}\n",
        )

        send_command_with_retry(self.acad, f"save\n{paths.dwg_path}\ny\n")
        send_command_with_retry(self.acad, f"Export\n{paths.sat_path}\ny\nALL\n\n")

        for _ in range(3):
            if os.path.exists(paths.sat_path) and os.path.exists(paths.dwg_path):
                break
            send_command_with_retry(self.acad, f"save\n{paths.dwg_path}\ny\n")
            send_command_with_retry(self.acad, f"Export\n{paths.sat_path}\ny\nALL\n\n")
            time.sleep(1)

        if os.path.exists(paths.sat_path) and os.path.exists(paths.dwg_path):
            send_command_with_retry(self.acad, "close\n")
            time.sleep(2)
        else:
            print(f"❌ 最終仍未成功產生檔案：{paths.sat_path} 或 {paths.dwg_path}。")


# ---------------------------------------------------------------------------
# Public wrapper to maintain backward compatibility
# ---------------------------------------------------------------------------

def Build_model(sid_ang, mode: str = "stair", folder: str = "."):
    """Legacy wrapper for building a prism model."""
    sid_ang = [round(sid_ang[0], 2), round(sid_ang[1], 2), sid_ang[2]]
    paths = OutputPaths(folder)
    for p in [paths.sat_path, paths.dwg_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:  # pragma: no cover - file system access
                print(f"⚠️ 無法刪除舊檔案 {p}: {e}")
    builder = PrismBuilder(scale=1)
    builder.build(sid_ang, mode=mode, paths=paths)
    return 1, []


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sid_ang = [0.48, 0.98, 85]
    Build_model(sid_ang, mode="triangle", folder=".")
