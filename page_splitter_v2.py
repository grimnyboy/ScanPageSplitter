import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


class PageSplitter:
    def __init__(self, input_folder=".", output_folder="split_pages"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.debug_folder, exist_ok=True)

    # ============================================================
    # NEW CORE METHOD - FIND LIGHT VALLEY (BOOK GUTTER)
    # ============================================================
    def find_vertical_line(
        self,
        image,
        darkness_threshold=110,
        center_weight=0.0005,
        use_edges_fallback=True,
    ):
        """
        Търси светлата вертикална зона (гръб на книга)
        чрез минимален процент тъмни пиксели.
        """

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Търсим само в централната част (20% - 80%)
        search_start = int(width * 0.20)
        search_end = int(width * 0.80)

        vertical_darkness_ratio = []

        for x in range(search_start, search_end):
            column = gray[:, x]
            dark_pixels = np.sum(column < darkness_threshold)
            dark_ratio = dark_pixels / height
            vertical_darkness_ratio.append(dark_ratio)

        vertical_darkness_ratio = np.array(vertical_darkness_ratio)

        # --------------------------------------------------------
        # 1) SMOOTHING
        # --------------------------------------------------------
        smooth = cv2.GaussianBlur(
            vertical_darkness_ratio.reshape(1, -1),
            (1, 51),
            0
        ).flatten()

        # --------------------------------------------------------
        # 2) CENTER BIAS
        # --------------------------------------------------------
        center = len(smooth) / 2
        center_bias = np.abs(np.arange(len(smooth)) - center)

        score = smooth + center_weight * center_bias

        best_idx = np.argmin(score)
        split_x = search_start + best_idx

        confidence = 1 - smooth[best_idx]

        # --------------------------------------------------------
        # 3) OPTIONAL EDGE FALLBACK
        # --------------------------------------------------------
        if use_edges_fallback and confidence < 0.3:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            edge_strength = np.abs(sobel_x).mean(axis=0)

            edge_smooth = cv2.GaussianBlur(
                edge_strength.reshape(1, -1),
                (1, 51),
                0
            ).flatten()

            edge_center = len(edge_smooth) / 2
            edge_bias = np.abs(np.arange(len(edge_smooth)) - edge_center)

            edge_score = edge_smooth - 0.0003 * edge_bias
            best_idx = np.argmax(edge_score[search_start:search_end])
            split_x = search_start + best_idx
            confidence = 0.5  # fallback confidence

        # Darkness map (debug)
        darkness_map = np.zeros((height, width), dtype=np.uint8)
        for x in range(search_start, search_end):
            column = gray[:, x]
            darkness_map[:, x] = (column < darkness_threshold).astype(np.uint8) * 255

        return {
            "split_x": split_x,
            "confidence": confidence,
            "vertical_darkness_ratio": smooth,
            "search_start": search_start,
            "darkness_threshold": darkness_threshold,
            "darkness_map": darkness_map,
        }

    # ============================================================
    # VISUALIZATION
    # ============================================================
    def visualize_split_analysis(self, image, analysis, output_path):

        fig, axes = plt.subplots(3, 1, figsize=(15, 14))

        vis_image = image.copy()
        height, width = vis_image.shape[:2]
        split_x = analysis["split_x"]

        cv2.line(vis_image, (split_x, 0), (split_x, height), (0, 0, 255), 3)

        search_start = analysis["search_start"]
        search_end = search_start + len(analysis["vertical_darkness_ratio"])
        cv2.rectangle(vis_image, (search_start, 0), (search_end, height), (255, 255, 0), 2)

        axes[0].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Split Preview (Red=Split, Yellow=Search Zone)")
        axes[0].axis("off")

        axes[1].imshow(analysis["darkness_map"], cmap="hot", aspect="auto")
        axes[1].axvline(x=split_x, color="cyan", linestyle="--", linewidth=2)
        axes[1].set_title("Darkness Heatmap")

        x_coords = np.arange(len(analysis["vertical_darkness_ratio"])) + search_start
        axes[2].plot(x_coords, analysis["vertical_darkness_ratio"] * 100, "r-", linewidth=2)
        axes[2].axvline(x=split_x, color="green", linestyle="--", linewidth=3)
        axes[2].set_title("Smoothed Dark Pixel % (Looking for MINIMUM)")
        axes[2].set_ylabel("Dark Pixel %")
        axes[2].set_xlabel("X Position")

        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ============================================================
    # SPLIT IMAGE
    # ============================================================
    def split_image(self, image, split_x):
        left = image[:, :split_x]
        right = image[:, split_x:]
        return left, right

    # ============================================================
    # PROCESS SINGLE IMAGE
    # ============================================================
    def process_image(self, image_path):

        print("\n" + "=" * 70)
        print(f"📄 Обработка: {image_path}")
        print("=" * 70)

        img = cv2.imread(image_path)
        if img is None:
            print("❌ Грешка при зареждане")
            return

        height, width = img.shape[:2]
        aspect_ratio = width / height

        print(f"📐 Размер: {width}x{height}, AR={aspect_ratio:.2f}")

        if aspect_ratio < 1.3:
            print("ℹ️  Вероятно не е двустранично изображение.")
            return

        analysis = self.find_vertical_line(img)

        split_x = analysis["split_x"]
        confidence = analysis["confidence"]

        print(f"✂️  Split at X={split_x}px | Confidence={confidence:.2f}")

        base_name = Path(image_path).stem

        debug_path = os.path.join(self.debug_folder, f"{base_name}_analysis.png")
        self.visualize_split_analysis(img, analysis, debug_path)

        left, right = self.split_image(img, split_x)

        left_path = os.path.join(self.output_folder, f"{base_name}_left.png")
        right_path = os.path.join(self.output_folder, f"{base_name}_right.png")

        cv2.imwrite(left_path, left)
        cv2.imwrite(right_path, right)

        print(f"✅ Лява страница: {left_path}")
        print(f"✅ Дясна страница: {right_path}")

    # ============================================================
    # PROCESS ALL
    # ============================================================
    def process_all_images(self):

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        image_files = [
            f for f in os.listdir(self.input_folder)
            if Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            print("❌ Няма намерени изображения.")
            return

        print(f"\n🔎 Намерени {len(image_files)} изображения")

        for image_file in image_files:
            full_path = os.path.join(self.input_folder, image_file)
            self.process_image(full_path)

        print("\n✅ Готово!")


def main():
    print("=" * 70)
    print("PAGE SPLITTER V3 - Light Valley Detection")
    print("=" * 70)

    splitter = PageSplitter(input_folder=".", output_folder="split_pages")
    splitter.process_all_images()


if __name__ == "__main__":
    main()
