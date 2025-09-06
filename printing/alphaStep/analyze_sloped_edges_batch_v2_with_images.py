import os
import csv
import matplotlib.pyplot as plt
from analyze_sloped_edges_by_profile_v2 import analyze_sloped_edges_by_profile_v2
import sys

sys.stdout.reconfigure(encoding="utf-8")

def batch_process_folder(
    folder_path,
    output_csv_path,
    output_image_folder,
    px_size=0.56,
    z_scale=0.56,
    **kwargs,
):
    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_ext)
    ]

    if not image_files:
        print("⚠️ No image files found in folder.")
        return

    os.makedirs(output_image_folder, exist_ok=True)
    records = []
    for idx, image_path in enumerate(sorted(image_files)):
        print(
            f"🖼️ [{idx+1}/{len(image_files)}] Processing: {os.path.basename(image_path)}"
        )

        result = analyze_sloped_edges_by_profile_v2(
            image_path=image_path,
            px_size=px_size,
            z_scale=z_scale,
            display=True,
            **kwargs,
        )

        if result is None:
            continue

        segments, fig = result
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_image_folder, f"{base}_annotated.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        for i, seg in enumerate(segments):
            records.append(
                {
                    "filename": os.path.basename(image_path),
                    "structure_index": i + 1,
                    "upper_angle": seg["upper_angle"],
                    "lower_angle": seg["lower_angle"],
                    "r2_upper": seg["r2_upper"],
                    "r2_lower": seg["r2_lower"],
                }
            )

    if records:
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "structure_index",
                    "upper_angle",
                    "lower_angle",
                    "r2_upper",
                    "r2_lower",
                ],
            )
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        print(f"✅ Results saved to: {output_csv_path}")
    else:
        print("⚠️ No valid structures found.")

batch_process_folder(
    folder_path=r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB",
    output_csv_path=r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB\sloped_edges_summary.csv",
    output_image_folder=r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB\AnnotatedImages",
    px_size=0.56,
    z_scale=0.56,
    rdp_tolerance=10.0,
    smooth_window=51,
    peak_prominence=20,
    peak_distance=100,
    margin_ratio_x=0.1, 
    margin_ratio_y=0.05,
    fit_margin=10,
    min_r2=0.90,
)
