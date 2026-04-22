"""
Report generator — produces a multi-page Matplotlib PDF with:
  • Hourly traffic trends
  • Vehicle type distribution
  • Road usage statistics
  • Congestion heatmap
  • Speed vs congestion scatter
  • 3-hour prediction
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from database.db_manager import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

COLOR_MAP = {
    "car":        "#1D9E75",
    "motorcycle": "#378ADD",
    "truck":      "#EF9F27",
    "bus":        "#E24B4A",
    "bicycle":    "#888780",
}
CONG_COLORS = {
    "low":      "#1D9E75",
    "moderate": "#EF9F27",
    "high":     "#E85D24",
    "severe":   "#E24B4A",
}


class ReportGenerator:
    def __init__(self, db: DatabaseManager):
        self.db = db
        os.makedirs("output", exist_ok=True)

    def generate_pdf(self, path: str = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or f"output/traffic_report_{ts}.pdf"
        logger.info(f"Generating PDF report: {path}")

        with PdfPages(path) as pdf:
            self._page_title(pdf)
            self._page_hourly_trends(pdf)
            self._page_vehicle_distribution(pdf)
            self._page_road_usage(pdf)
            self._page_heatmap(pdf)
            self._page_prediction(pdf)

        logger.info(f"Report saved: {path}")
        return path

    # ------------------------------------------------------------------
    def _style(self):
        plt.rcParams.update({
            "figure.facecolor":  "#0F1117",
            "axes.facecolor":    "#161B22",
            "axes.edgecolor":    "#30363D",
            "axes.labelcolor":   "#C9D1D9",
            "xtick.color":       "#8B949E",
            "ytick.color":       "#8B949E",
            "text.color":        "#C9D1D9",
            "grid.color":        "#21262D",
            "grid.linewidth":    0.5,
            "font.family":       "DejaVu Sans",
        })

    def _page_title(self, pdf):
        self._style()
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_facecolor("#0F1117")
        fig.patch.set_facecolor("#0F1117")
        ax.axis("off")

        stats = self.db.get_summary_stats()
        ax.text(0.5, 0.72, "Smart Traffic Analysis System",
                ha="center", va="center", fontsize=28, fontweight="bold",
                color="#1D9E75", transform=ax.transAxes)
        ax.text(0.5, 0.62, "Traffic Flow Report",
                ha="center", va="center", fontsize=18, color="#8B949E",
                transform=ax.transAxes)
        ax.text(0.5, 0.52, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}",
                ha="center", va="center", fontsize=13, color="#8B949E",
                transform=ax.transAxes)

        # Summary boxes
        items = [
            ("Total Vehicles Today", f"{int(stats.get('total_vehicles_today', 0)):,}"),
            ("Average Speed",        f"{stats.get('avg_speed', 0):.1f} km/h"),
            ("Avg Congestion",       f"{stats.get('avg_congestion', 0):.0f}/100"),
            ("Active Incidents",     str(int(stats.get('active_incidents', 0)))),
        ]
        for i, (label, val) in enumerate(items):
            x = 0.1 + i * 0.22
            ax.add_patch(plt.Rectangle((x, 0.28), 0.18, 0.14,
                         transform=ax.transAxes, color="#161B22", zorder=2))
            ax.text(x + 0.09, 0.41, val,
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="#1D9E75", transform=ax.transAxes, zorder=3)
            ax.text(x + 0.09, 0.31, label,
                    ha="center", va="center", fontsize=8, color="#8B949E",
                    transform=ax.transAxes, zorder=3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    def _page_hourly_trends(self, pdf):
        self._style()
        data = self.db.get_hourly_trend(hours=24)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), facecolor="#0F1117")
        fig.suptitle("Hourly Traffic Trends", fontsize=16, color="#C9D1D9", y=0.97)

        if data:
            roads = list(set(r["road_name"] for r in data))
            road_colors = plt.cm.get_cmap("tab10", len(roads))

            ax = axes[0]
            ax.set_facecolor("#161B22")
            for i, road in enumerate(roads):
                rd = [r for r in data if r["road_name"] == road]
                if rd:
                    hours = [r["hour"] for r in rd]
                    counts = [r["total_vehicles"] or 0 for r in rd]
                    ax.plot(hours, counts, marker="o", markersize=3,
                            label=road, color=road_colors(i), linewidth=1.5)
            ax.set_ylabel("Vehicle Count")
            ax.set_title("Vehicles per Hour by Road", color="#8B949E", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.4)
            ax.tick_params(axis="x", rotation=30)

            ax2 = axes[1]
            ax2.set_facecolor("#161B22")
            for i, road in enumerate(roads):
                rd = [r for r in data if r["road_name"] == road]
                if rd:
                    hours = [r["hour"] for r in rd]
                    cong = [r["avg_congestion"] or 0 for r in rd]
                    ax2.plot(hours, cong, marker="s", markersize=3,
                             label=road, color=road_colors(i), linewidth=1.5)
            ax2.set_ylabel("Congestion Score (0–100)")
            ax2.set_title("Average Congestion Score per Hour", color="#8B949E", fontsize=10)
            ax2.set_ylim(0, 100)
            ax2.axhline(60, color="#EF9F27", linewidth=0.8, linestyle="--", alpha=0.7, label="High threshold")
            ax2.axhline(80, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.7, label="Severe threshold")
            ax2.legend(fontsize=8, loc="upper left")
            ax2.grid(True, alpha=0.4)
            ax2.tick_params(axis="x", rotation=30)
        else:
            for ax in axes:
                ax.set_facecolor("#161B22")
                ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                        transform=ax.transAxes, color="#8B949E")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    def _page_vehicle_distribution(self, pdf):
        self._style()
        totals = self.db.get_vehicle_type_totals(hours=24)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5), facecolor="#0F1117")
        fig.suptitle("Vehicle Type Distribution", fontsize=16, color="#C9D1D9", y=0.97)

        if totals and any(totals.values()):
            labels = [k for k, v in totals.items() if v and int(v) > 0]
            values = [int(totals[k]) for k in labels]
            colors = [COLOR_MAP.get(k, "#888780") for k in labels]

            # Donut
            ax1.set_facecolor("#161B22")
            wedges, texts, autotexts = ax1.pie(
                values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"width": 0.55, "edgecolor": "#0F1117", "linewidth": 2},
            )
            for t in texts:
                t.set_color("#C9D1D9"); t.set_fontsize(9)
            for at in autotexts:
                at.set_color("#0F1117"); at.set_fontsize(8); at.set_fontweight("bold")
            ax1.set_title("Share by vehicle type", color="#8B949E", fontsize=10, pad=15)

            # Bar
            ax2.set_facecolor("#161B22")
            bars = ax2.barh(labels, values, color=colors, height=0.55,
                            edgecolor="#0F1117", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax2.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{val:,}", va="center", fontsize=9, color="#C9D1D9")
            ax2.set_xlabel("Count")
            ax2.set_title("Total count per type (24h)", color="#8B949E", fontsize=10)
            ax2.grid(axis="x", alpha=0.4)
            ax2.invert_yaxis()
        else:
            for ax in (ax1, ax2):
                ax.set_facecolor("#161B22")
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="#8B949E")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    def _page_road_usage(self, pdf):
        self._style()
        data = self.db.get_road_usage_stats()
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), facecolor="#0F1117")
        fig.suptitle("Road Usage Statistics", fontsize=16, color="#C9D1D9", y=0.97)

        if data:
            roads = [r["road_name"] for r in data]
            avg_cong = [float(r["avg_congestion"] or 0) for r in data]
            avg_speed = [float(r["avg_speed"] or 0) for r in data]

            colors = [
                "#E24B4A" if c >= 80 else "#EF9F27" if c >= 60 else "#1D9E75"
                for c in avg_cong
            ]

            ax1 = axes[0]
            ax1.set_facecolor("#161B22")
            bars = ax1.barh(roads, avg_cong, color=colors, height=0.55)
            ax1.axvline(60, color="#EF9F27", linewidth=1, linestyle="--", alpha=0.7)
            ax1.axvline(80, color="#E24B4A", linewidth=1, linestyle="--", alpha=0.7)
            for bar, v in zip(bars, avg_cong):
                ax1.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{v:.0f}", va="center", fontsize=9, color="#C9D1D9")
            ax1.set_xlabel("Avg Congestion Score (0–100)")
            ax1.set_xlim(0, 105)
            ax1.set_title("Average congestion by road", color="#8B949E", fontsize=10)
            ax1.grid(axis="x", alpha=0.4)
            ax1.invert_yaxis()

            ax2 = axes[1]
            ax2.set_facecolor("#161B22")
            bars2 = ax2.barh(roads, avg_speed, color="#378ADD", height=0.55)
            for bar, v in zip(bars2, avg_speed):
                ax2.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{v:.1f}", va="center", fontsize=9, color="#C9D1D9")
            ax2.set_xlabel("Average Speed (km/h)")
            ax2.set_title("Average speed by road", color="#8B949E", fontsize=10)
            ax2.grid(axis="x", alpha=0.4)
            ax2.invert_yaxis()
        else:
            for ax in axes:
                ax.set_facecolor("#161B22")
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="#8B949E")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    def _page_heatmap(self, pdf):
        self._style()
        roads = ["NH-75", "Ring Road N", "SH-12 South", "City Bypass", "Connector A", "East Connector"]
        matrix_data = []
        road_labels = []
        for road in roads:
            peaks = self.db.get_peak_hours(road_name=road)
            if peaks:
                hour_vals = {int(r["hour"]): float(r["avg_congestion"] or 0) for r in peaks}
                row = [hour_vals.get(h, 0) for h in range(24)]
                matrix_data.append(row)
                road_labels.append(road)

        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="#0F1117")
        fig.suptitle("Congestion Heatmap — Hour × Road", fontsize=16, color="#C9D1D9", y=0.97)
        ax.set_facecolor("#161B22")

        if matrix_data:
            M = np.array(matrix_data)
            im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
            ax.set_xticks(range(24))
            ax.set_xticklabels(
                [f"{h:02d}:00" for h in range(24)],
                fontsize=7, rotation=45, ha="right"
            )
            ax.set_yticks(range(len(road_labels)))
            ax.set_yticklabels(road_labels, fontsize=9)
            ax.set_xlabel("Hour of day")
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Congestion Score", color="#C9D1D9", fontsize=9)
            cbar.ax.yaxis.set_tick_params(color="#C9D1D9")

            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if M[i,j] > 50 else "#0F1117")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#8B949E")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    def _page_prediction(self, pdf):
        self._style()
        road = "NH-75"
        preds = self.db.get_congestion_prediction(road, look_ahead_hours=6)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), facecolor="#0F1117")
        fig.suptitle(f"Congestion Prediction — {road}", fontsize=16, color="#C9D1D9", y=0.97)

        if preds:
            times = [r["forecast_time"] for r in preds]
            cong = [float(r["predicted_congestion"] or 40) for r in preds]
            speed = [float(r["predicted_speed"] or 30) for r in preds]

            ax1.set_facecolor("#161B22")
            ax1.fill_between(range(len(times)), cong, alpha=0.25, color="#E24B4A")
            ax1.plot(range(len(times)), cong, color="#E24B4A", linewidth=2, marker="o", markersize=5)
            ax1.axhline(60, color="#EF9F27", linewidth=0.8, linestyle="--", alpha=0.6, label="High")
            ax1.axhline(80, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.6, label="Severe")
            ax1.set_xticks(range(len(times)))
            ax1.set_xticklabels(
                [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in times],
                fontsize=9, rotation=20
            )
            ax1.set_ylabel("Predicted Congestion Score")
            ax1.set_ylim(0, 105)
            ax1.set_title("Congestion forecast (next 6 hours)", color="#8B949E", fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.4)

            ax2.set_facecolor("#161B22")
            ax2.fill_between(range(len(times)), speed, alpha=0.25, color="#1D9E75")
            ax2.plot(range(len(times)), speed, color="#1D9E75", linewidth=2, marker="^", markersize=5)
            ax2.set_xticks(range(len(times)))
            ax2.set_xticklabels(
                [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t) for t in times],
                fontsize=9, rotation=20
            )
            ax2.set_ylabel("Predicted Avg Speed (km/h)")
            ax2.set_title("Speed forecast (next 6 hours)", color="#8B949E", fontsize=10)
            ax2.grid(True, alpha=0.4)

            # Suggested route annotation
            max_cong = max(cong)
            if max_cong >= 80:
                note = "Recommendation: Use Ring Road North → City Bypass to avoid severe congestion."
                ax1.text(0.01, 0.92, note, transform=ax1.transAxes,
                         fontsize=8, color="#EF9F27",
                         bbox={"boxstyle": "round,pad=0.3", "facecolor": "#161B22", "edgecolor": "#EF9F27", "alpha": 0.8})
        else:
            for ax in (ax1, ax2):
                ax.set_facecolor("#161B22")
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="#8B949E")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
