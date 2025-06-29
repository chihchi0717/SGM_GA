# AutoCAD prism structure generator with base support

import math
import time
import sys
import warnings
import os
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
    sat_name: str  # = "prism_sat_file-print_0.46_0.95_85.SAT"
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

    def __init__(
        self, scale: float = 1.0, pixel_size: int = 22, sleep_time: float = 0.2
    ):
        self.scale = scale
        self.pixel_size = pixel_size
        self.sleep_time = sleep_time

        # Launch AutoCAD and create a new document
        self.acad = retry_autocad_call(lambda: Autocad(create_if_not_exists=True))
        retry_autocad_call(lambda: self.acad.app.Documents.Add())
        time.sleep(self.sleep_time)

        # Configure units
        try:
            self.acad.ActiveDocument.SendCommand("-UNITS\n2\n4\n1\n4\n0\nY\n\n")
            time.sleep(self.sleep_time)
        except Exception as e:  # pragma: no cover - direct COM calls
            print(f"⚠️ 設定 UNITS 命令失敗：{e}")
            raise
        send_command_with_retry(self.acad, "FACETRES\n10\n")

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
        side_c = math.sqrt(
            side_a**2
            + side_b**2
            - 2 * side_a * side_b * math.cos(math.radians(angle_B))
        )
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
        pos, pos1, pos2 = [(-22, bottom), (-22, top)], [], []
        while True:
            equ_y1 = equ_ac(current_x1)
            equ_y2 = equ_bc(current_x2)
            real_y1 = math.ceil(equ_y1 / self.pixel_size) * self.pixel_size
            real_y2 = math.floor(equ_y2 / self.pixel_size) * self.pixel_size

            if real_y2 != current_y2 and current_y1 != real_y2:
                current_x2 -= self.pixel_size
                pos.append((current_x2, current_y2))
                pos.append((current_x2, real_y2))
                pos2.extend([(current_x2, current_y2), (current_x2, real_y2)])
            elif real_y2 == real_y1:
                break
            else:
                pos.append((current_x2 - self.pixel_size, real_y2))
                pos.append((current_x2, real_y2))
                pos2.extend(
                    [(current_x2 - self.pixel_size, real_y2), (current_x2, real_y2)]
                )

            if real_y1 != current_y1 and current_y1 != real_y2:
                current_x1 -= self.pixel_size
                pos.append((current_x1, current_y1))
                pos.append((current_x1, real_y1))
                pos1.extend([(current_x1, current_y1), (current_x1, real_y1)])
            elif real_y1 == real_y2:
                break
            else:
                pos.append((current_x1 - self.pixel_size, real_y1))
                pos.append((current_x1, real_y1))
                pos1.extend(
                    [(current_x1 - self.pixel_size, real_y1), (current_x1, real_y1)]
                )

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
            APoint(-22 * self.scale, bottom * self.scale),
            APoint(-22 * self.scale, top * self.scale),
        )
        self.acad.model.AddLine(
            APoint(pos1[-1][0] * self.scale, pos1[-1][1] * self.scale),
            APoint(pos2[-1][0] * self.scale, pos2[-1][1] * self.scale),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        sid_ang,
        mode: str,
        paths: OutputPaths,
        sub_length_x,
        base_length_y,
        sub_thickness,
        base_length_x,
        base_thickness,
        fillet=1,
    ) -> None:
        """Build a prism structure in AutoCAD."""
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
            send_command_with_retry(
                self.acad,
                f"_.ZOOM\nE\n\n",
            )
            if fillet == 0:

                send_command_with_retry(self.acad, "SELECT\nALL\n\n_JOIN\n\n")
                send_command_with_retry(self.acad, "ZOOM\nE\n\n")
                send_command_with_retry(self.acad, f"-BOUNDARY\n{Ix},{Iy}\n\n")
                send_command_with_retry(self.acad, f"_EXTRUDE\nL\n\n{sub_thickness}\n")
                send_command_with_retry(self.acad, "UNION\nALL\n\n")

                rows, columns = int(base_length_y / side_a), 1
                row_spacing = side_a * self.scale * (rows - 1)
                column_spacing = 1
                send_command_with_retry(
                    self.acad,
                    f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
                )

                send_command_with_retry(self.acad, "Explode\nALL\n\n")
                send_command_with_retry(self.acad, "UNION\nALL\n\n")

            if fillet == 1:
                radius = 0.088  # 0.05
                x = round(Cx * self.scale, 1)
                y = round(Cy * self.scale, 1)
                corner_x1 = x + 0.5
                corner_y1 = y - 0.5
                corner_x2 = x - 0.5
                corner_y2 = y + 0.5

                send_command_with_retry(
                    self.acad,
                    f"FILLET\nRadius\n{radius}\nC\n{corner_x1},{corner_y1}\n{corner_x2},{corner_y2}\n",
                )
                rows, columns = int(base_length_y / side_a), 1  # 30, 1
                row_spacing = side_a * self.scale * (rows - 1)
                column_spacing = 1
                send_command_with_retry(
                    self.acad,
                    f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
                )

                send_command_with_retry(self.acad, "Explode\nALL\n\n")
                send_command_with_retry(self.acad, "ZOOM\nE\n")

                send_command_with_retry(self.acad, "SELECT\nALL\n\nJOIN\nALL\n\n")
                send_command_with_retry(self.acad, "SELECT\nALL\n\n_JOIN\n\n")
                send_command_with_retry(self.acad, "ZOOM\nE\n\n")

                # send_command_with_retry(self.acad, f"-BOUNDARY\n{Ix},{Iy}\n\n")
                send_command_with_retry(
                    self.acad, f"_EXTRUDE\nALL\n\n{sub_thickness}\n"
                )
                time.sleep(self.sleep_time)
                send_command_with_retry(self.acad, "UNION\nALL\n\n")

            if fillet == 2:
                radius = 0.066  # 0.05
                x = round(Cx * self.scale, 1)
                y = round(Cy * self.scale, 1)
                corner_x1 = x + 0.5
                corner_y1 = y - 0.5
                corner_x2 = x - 0.5
                corner_y2 = y + 0.5

                send_command_with_retry(
                    self.acad,
                    f"FILLET\nRadius\n{radius}\nC\n{corner_x1},{corner_y1}\n{corner_x2},{corner_y2}\n",
                )
                rows, columns = 2, 1
                row_spacing = side_a * self.scale * (rows - 1)
                column_spacing = 1
                send_command_with_retry(
                    self.acad,
                    f"ARRAY\nALL\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
                )

                send_command_with_retry(self.acad, "Explode\nALL\n\n")
                send_command_with_retry(
                    self.acad,
                    f"_.ZOOM\nE\n\n",
                )
                x = round((Cx / 2) * self.scale, 1)
                y = round((Cy / 2) * self.scale, 1)
                print(f"Fillet corner at: ({x}, {y})")
                corner_x3 = 0.4
                corner_y3 = 0.2
                corner_x4 = 0.1
                corner_y4 = 0.7
                equ_bc(corner_x3)

                send_command_with_retry(
                    self.acad,
                    f"FILLET\nRadius\n{radius}\nC\n{corner_x3},{corner_y3}\n{corner_x4},{corner_y4}\n",
                )
                rows, columns = int(base_length_y / side_a), 1
                row_spacing = side_a * self.scale * (rows - 1)
                column_spacing = 1
                send_command_with_retry(
                    self.acad,
                    f"ARRAY\nC\n{Cx+0.05},{top+0.05}\n{0},{top*1.8+0.05}\n\nR\nCOL\n{columns}\nT\n{column_spacing}\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
                )
                send_command_with_retry(self.acad, "Explode\nALL\n\n")

                send_command_with_retry(self.acad, "ZOOM\nE\n")
                send_command_with_retry(
                    self.acad, f"TRIM\n{0.1},{top*1.5}\n{0.1},{top*30}\n\n"
                )
                send_command_with_retry(self.acad, "SELECT\nALL\n\nJOIN\nALL\n\n")
                send_command_with_retry(self.acad, "SELECT\nALL\n\n_JOIN\n\n")
                send_command_with_retry(self.acad, "ZOOM\nE\n\n")

                send_command_with_retry(self.acad, f"-BOUNDARY\n{Ix},{Iy}\n\n")
                send_command_with_retry(self.acad, f"_EXTRUDE\nL\n\n{sub_thickness}\n")
                time.sleep(self.sleep_time)
                send_command_with_retry(self.acad, "UNION\nALL\n\n")

        elif mode == "stair":
            self._draw_stair(equ_ac, equ_bc, bottom, top)
        else:
            raise ValueError("mode must be 'stair' or 'triangle'")

        # send_command_with_retry(self.acad, "SELECT\nALL\n\n_JOIN\n\n")
        # send_command_with_retry(self.acad, "ZOOM\nE\n\n")
        # send_command_with_retry(
        #     self.acad,
        #     f"-BOUNDARY\n{round(Ix * self.scale, 4)},{round(Iy * self.scale, 4)}\n\n",
        # )

        # send_command_with_retry(self.acad, f"_EXTRUDE\nL\n\n{base_length_y}\n")
        # send_command_with_retry(self.acad, "UNION\nALL\n\n")

        # rows = int(base_length_y / (side_a * self.scale))
        # row_spacing = side_a * self.scale * (rows - 1)
        # send_command_with_retry(
        #     self.acad,
        #     f"ARRAY\nALL\n\nR\nCOL\n1\nT\n1\nR\n{rows}\nT\n{row_spacing}\n0\nX\n",
        # )
        # time.sleep(self.sleep_time)
        # send_command_with_retry(self.acad, "Explode\nALL\n\n")
        # send_command_with_retry(self.acad, "UNION\nALL\n\n")

        self._add_substrate(paths, sub_length_x, base_length_y, sub_thickness)
        self._add_base(paths, Cx, base_length_x, base_length_y, base_thickness)

        send_command_with_retry(self.acad, f"Export\n{paths.sat_path}\nY\nALL\n\n")
        send_command_with_retry(self.acad, f"save\n{paths.dwg_path}\nY\n")

    # ------------------------------------------------------------------
    def _add_substrate(self, paths: OutputPaths, x, y, z):
        start_base = APoint(0, 0, 0)
        sub_length_x = x  # 1.1
        sub_length_y = y  # 30  # 55
        sub_thickness = z  # 27  # 45
        send_command_with_retry(
            self.acad,
            f"_BOX\n{start_base.x - sub_length_x},{start_base.y},{start_base.z}\n{start_base.x},{start_base.y + sub_length_y},{start_base.z + sub_thickness}\n",
        )

    def _add_base(self, paths: OutputPaths, Cx: float, x, y, z):
        base_length_x = x  # 10
        base_length_y = y  # 30  # 55
        base_thickness = z  # 15
        start_base = APoint(base_length_x / 2, 0, 0)
        send_command_with_retry(
            self.acad,
            f"_BOX\n{start_base.x - base_length_x},{start_base.y},{start_base.z}\n{start_base.x * self.scale},{start_base.y + base_length_y},{start_base.z - base_thickness}\n",
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main():
    s = 1  # mm
    sid_ang = [0.6 * s, 1.2 * s, 79]
    folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202506\0628"
    sat_name = os.path.join(folder, "prism_sat_file-print_0.6_1.2_79.SAT")
    paths = OutputPaths(folder=folder, sat_name=sat_name)

    builder = PrismBuilder(scale=1)
    builder.build(
        sid_ang,
        mode="triangle",
        paths=paths,
        sub_length_x=0.6,
        base_length_y=38,
        sub_thickness=15,
        base_length_x=11,
        base_thickness=5,
        fillet=0,
    )


if __name__ == "__main__":
    main()
