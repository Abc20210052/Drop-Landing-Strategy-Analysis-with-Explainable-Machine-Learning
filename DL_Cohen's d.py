import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

output_root = 'Biomechanics_Feature_Analysis'
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"Created main output directory: {output_root}")


def save_figure(fig, filename, dpi=300):
    os.makedirs(os.path.dirname(os.path.join(output_root, filename)), exist_ok=True)

    png_path = os.path.join(output_root, f"{filename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved PNG: {png_path}")

    svg_path = os.path.join(output_root, f"{filename}.svg")
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  Saved SVG: {svg_path}")

    pdf_path = os.path.join(output_root, f"{filename}.pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved PDF: {pdf_path}")


class BiomechanicsFeatureAnalyzer:
    def __init__(self):
        self.file_paths = [
            r'F:\DL_40.csv',
            r'F:\DL_80.csv',
            r'F:\DL_P40.csv',
            r'F:\DL_P80.csv'
        ]
        self.class_names = ['40', '80', 'F40', 'F80']

        self.feature_groups = {
            'Hip Angle': (0, 201),
            'Hip Moment': (201, 402),
            'Knee Angle': (402, 603),
            'Knee Moment': (603, 804),
            'Ankle Angle': (804, 1005),
            'Ankle Moment': (1005, 1206),
            'Vertical GRF': (1206, 1407)
        }

        self.group_names = ['Hip Angle', 'Hip Moment', 'Knee Angle',
                            'Knee Moment', 'Ankle Angle', 'Ankle Moment', 'Vertical GRF']
        self.n_subjects = 20
        self.n_repeats = 3
        self.n_features = 1407

        self.original_time_points = np.arange(201)
        self.target_time_points = np.linspace(0, 100, 101)

        self.all_analysis_results = {}

    def load_and_prepare_data(self):
        all_data = []
        all_labels = []
        all_groups = []

        for i, file_path in enumerate(self.file_paths):
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                data = df.values
                print(f"Load {self.class_names[i]}: {data.shape}")
            else:
                np.random.seed(42 + i)
                data = np.random.randn(60, self.n_features)

                for group_idx, (group_name, (start, end)) in enumerate(self.feature_groups.items()):
                    time = np.linspace(0, 4 * np.pi, 201)

                    if 'angle' in group_name:
                        pattern = np.sin(time) * 0.8
                    elif 'moment' in group_name:
                        pattern = np.sin(time) * np.cos(time) * 0.6
                    else:
                        pattern = np.exp(-((time - 2 * np.pi) / 1.5) ** 2) + 0.5 * np.exp(
                            -((time - 4 * np.pi) / 2) ** 2)

                    if '40' in self.class_names[i]:
                        data[:, start:end] += pattern * 0.5
                    else:
                        data[:, start:end] += pattern * 0.8 + group_idx * 0.1

                print(f"Create synthetic data {self.class_names[i]}: {data.shape}")

            labels = np.full(data.shape[0], i)
            groups = []
            for person in range(self.n_subjects):
                groups.extend([person + i * 100] * self.n_repeats)

            all_data.append(data)
            all_labels.append(labels)
            all_groups.extend(groups)

        X = np.vstack(all_data)
        y = np.hstack(all_labels)
        groups = np.array(all_groups)

        print(f"\nTotal data dimension: {X.shape}")
        print(f"Feature groups: {len(self.feature_groups)} groups, 201 time points each")
        print(f"Will be resampled to: 101 time points (0-100%)")

        return X, y, groups

    def resample_feature_group(self, feature_data):
        n_samples = feature_data.shape[0]
        resampled_data = np.zeros((n_samples, 101))

        for i in range(n_samples):
            f = interp1d(self.original_time_points, feature_data[i, :],
                         kind='cubic', fill_value='extrapolate')
            resampled_data[i, :] = f(np.linspace(0, 200, 101))

        return resampled_data

    def resample_all_features(self, X):
        n_samples = X.shape[0]
        resampled_X = np.zeros((n_samples, 7 * 101))

        for group_idx, (group_name, (start, end)) in enumerate(self.feature_groups.items()):
            group_data = X[:, start:end]
            resampled_group = self.resample_feature_group(group_data)
            output_start = group_idx * 101
            output_end = (group_idx + 1) * 101
            resampled_X[:, output_start:output_end] = resampled_group
            print(f"  Resampled {group_name}: {group_data.shape} -> {resampled_group.shape}")

        return resampled_X

    def analyze_pair_comparison(self, class_pair, X, y):
        print(f"\n{'=' * 60}")
        print(f"Analysis: {class_pair[0]} vs {class_pair[1]}")
        print('=' * 60)

        idx1 = self.class_names.index(class_pair[0])
        idx2 = self.class_names.index(class_pair[1])
        mask = (y == idx1) | (y == idx2)
        X_pair = X[mask]
        y_pair = y[mask]

        print(f"Original data shape: {X_pair.shape}")
        print(f"Class {class_pair[0]} samples: {sum(y_pair == idx1)}")
        print(f"Class {class_pair[1]} samples: {sum(y_pair == idx2)}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pair)

        print("\nResampling features from 201 to 101 time points...")
        X_resampled = self.resample_all_features(X_scaled)
        print(f"Resampled data shape: {X_resampled.shape}")

        X_class0 = X_resampled[y_pair == idx1]
        X_class1 = X_resampled[y_pair == idx2]

        mean_class0 = np.mean(X_class0, axis=0)
        mean_class1 = np.mean(X_class1, axis=0)
        std_class0 = np.std(X_class0, axis=0)
        std_class1 = np.std(X_class1, axis=0)

        pooled_std = np.sqrt((std_class0 ** 2 + std_class1 ** 2) / 2)
        normalized_diff = (mean_class1 - mean_class0) / (pooled_std + 1e-10)

        abs_diff = np.abs(mean_class1 - mean_class0)

        abs_diff_matrix = abs_diff.reshape(7, 101)
        norm_diff_matrix = normalized_diff.reshape(7, 101)

        print(f"\nDifference statistics for {class_pair[0]} vs {class_pair[1]}:")
        print(f"  Mean absolute diff: {np.mean(abs_diff):.4f}")
        print(f"  Max absolute diff: {np.max(abs_diff):.4f}")
        print(f"  Min absolute diff: {np.min(abs_diff):.4f}")

        return {
            'class_pair': class_pair,
            'abs_diff_matrix': abs_diff_matrix,
            'norm_diff_matrix': norm_diff_matrix,
            'X_resampled': X_resampled,
            'y_pair': y_pair
        }

    def plot_between_class_difference_heatmap(self, analysis_result):
        class_pair = analysis_result['class_pair']
        abs_diff_matrix = analysis_result['abs_diff_matrix']

        fig, ax = plt.subplots(figsize=(14, 7))

        im = ax.imshow(abs_diff_matrix, aspect='auto', cmap='hot_r',
                       extent=[0, 100, 6.5, -0.5], vmin=0, interpolation='nearest')

        ax.set_title(f'{class_pair[0]} vs {class_pair[1]}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Landing Phase (%)', fontsize=14)
        ax.set_ylabel('Biomechanical Feature', fontsize=14)

        ax.set_yticks(range(7))
        ax.set_yticklabels(self.group_names, fontsize=16)

        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
                           fontsize=16)

        for i in range(1, 7):
            ax.axhline(y=i - 0.5, color='white', linestyle='-', linewidth=1, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Absolute Difference', fontsize=18)
        cbar.ax.tick_params(labelsize=11)

        plt.tight_layout()

        filename = f"heatmap_{class_pair[0]}_vs_{class_pair[1]}"
        save_figure(fig, filename)
        plt.close()

        return fig

    def plot_combined_heatmap_simple(self):
        print("\n" + "=" * 80)
        print("Plotting combined heatmap (simple拼接)")
        print("=" * 80)

        if len(self.all_analysis_results) < 4:
            print("Error: Need 4 comparison results")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes_flat = axes.flatten()

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            ('40', '80'): 'Task 1 vs Task 2',
            ('F40', 'F80'): 'Task 3 vs Task 4',
            ('40', 'F40'): 'Task 1 vs Task 3',
            ('80', 'F80'): 'Task 2 vs Task 4'
        }

        for idx, (ax, (class_pair, analysis_result), label) in enumerate(zip(
                axes_flat, self.all_analysis_results.items(), subplot_labels)):

            abs_diff_matrix = analysis_result['abs_diff_matrix']

            print(f"Plotting {class_pair}: matrix shape {abs_diff_matrix.shape}, "
                  f"value range [{np.min(abs_diff_matrix):.3f}, {np.max(abs_diff_matrix):.3f}]")

            im = ax.imshow(abs_diff_matrix, aspect='auto', cmap='hot_r',
                           extent=[0, 100, 6.5, -0.5], interpolation='nearest')

            display_title = title_mapping.get(class_pair, f"{class_pair[0]} vs {class_pair[1]}")
            ax.set_title(display_title, fontsize=14, fontweight='bold', pad=15)

            ax.set_xlabel('Landing Phase (%)', fontsize=18)
            ax.set_ylabel('Biomechanical Feature', fontsize=18)

            ax.set_yticks(range(7))
            ax.set_yticklabels(self.group_names, fontsize=16)

            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=16)

            for i in range(1, 7):
                ax.axhline(y=i - 0.5, color='white', linestyle='-', linewidth=1, alpha=0.5)

            cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label('Absolute Difference', fontsize=18)
            cbar.ax.tick_params(labelsize=10)

            ax.annotate(label, xy=(-0.08, 1.02), xycoords='axes fraction',
                        fontsize=18, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.25, wspace=0.25)

        plt.show()

        print("\nSaving figures...")

        png_path = os.path.join(output_root, "combined_heatmap_simple.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved PNG: {png_path}")

        svg_path = os.path.join(output_root, "combined_heatmap_simple.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        print(f"  Saved SVG: {svg_path}")

        pdf_path = os.path.join(output_root, "combined_heatmap_simple.pdf")
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"  Saved PDF: {pdf_path}")

        plt.close(fig)
        print(f"\nCombined heatmap saved")

    def plot_time_importance_for_each_feature_group(self, analysis_result):
        class_pair = analysis_result['class_pair']
        abs_diff_matrix = analysis_result['abs_diff_matrix']

        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        axes = axes.flatten()

        for i, (group_name, group_label) in enumerate(zip(self.feature_groups.keys(), self.group_names)):
            ax = axes[i]
            time_points = self.target_time_points
            diff_values = abs_diff_matrix[i]

            mean_diff = np.mean(diff_values)
            std_diff = np.std(diff_values)
            threshold = mean_diff + std_diff
            significant_regions = diff_values > threshold

            ax.plot(time_points, diff_values, 'b-', linewidth=2.5)
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, linewidth=2)

            if np.any(significant_regions):
                start_idx = None
                for j, is_sig in enumerate(significant_regions):
                    if is_sig and start_idx is None:
                        start_idx = j
                    elif not is_sig and start_idx is not None:
                        ax.axvspan(time_points[start_idx], time_points[j - 1],
                                   alpha=0.3, color='yellow')
                        start_idx = None

                if start_idx is not None:
                    ax.axvspan(time_points[start_idx], time_points[-1],
                               alpha=0.3, color='yellow')

            ax.set_title(f'{group_label}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Landing Phase (%)', fontsize=12)
            ax.set_ylabel('Absolute Difference', fontsize=12)
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.axvline(x=50, color='green', linestyle=':', alpha=0.7, linewidth=1.5)

        if len(self.feature_groups) < len(axes):
            axes[-1].axis('off')

        plt.suptitle(f'{class_pair[0]} vs {class_pair[1]}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        filename = f"time_importance_{class_pair[0]}_vs_{class_pair[1]}"
        save_figure(fig, filename)
        plt.close()

        return fig

    def plot_normalized_difference_comparison(self, analysis_result):
        class_pair = analysis_result['class_pair']
        norm_diff_matrix = analysis_result['norm_diff_matrix']

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, 7))

        for i, (group_label, color) in enumerate(zip(self.group_names, colors)):
            time_points = self.target_time_points
            norm_values = norm_diff_matrix[i]
            smoothed = gaussian_filter1d(norm_values, sigma=2)
            ax.plot(time_points, smoothed, label=group_label,
                    color=color, linewidth=2.5, alpha=0.9)

        ax.set_title(f'{class_pair[0]} vs {class_pair[1]}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Landing Phase (%)', fontsize=18)
        ax.set_ylabel("Normalized Difference (Cohen's d)", fontsize=18)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        effect_thresholds = [
            (0.2, 'gray', '--', 0.5),
            (0.5, 'gray', '--', 0.7),
            (0.8, 'gray', '--', 0.9)
        ]

        for threshold, color, linestyle, alpha in effect_thresholds:
            ax.axhline(y=threshold, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)
            ax.axhline(y=-threshold, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)

        ax.set_xlim(0, 100)
        all_norm_values = norm_diff_matrix.flatten()
        y_max = max(np.abs(all_norm_values).max() * 1.2, 1.0)
        ax.set_ylim(-y_max, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)

        plt.tight_layout()

        filename = f"normalized_diff_{class_pair[0]}_vs_{class_pair[1]}"
        save_figure(fig, filename)
        plt.close()

        return fig

    def plot_combined_normalized_difference(self):
        print("\n" + "=" * 80)
        print("Plotting combined normalized difference comparison")
        print("=" * 80)

        if len(self.all_analysis_results) < 4:
            print("Error: Need 4 comparison results")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes_flat = axes.flatten()

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            ('40', '80'): 'Task 1 vs Task 2',
            ('F40', 'F80'): 'Task 3 vs Task 4',
            ('40', 'F40'): 'Task 1 vs Task 3',
            ('80', 'F80'): 'Task 2 vs Task 4'
        }

        colors = plt.cm.Set2(np.linspace(0, 1, 7))

        for idx, (ax, (class_pair, analysis_result), label) in enumerate(zip(
                axes_flat, self.all_analysis_results.items(), subplot_labels)):

            norm_diff_matrix = analysis_result['norm_diff_matrix']

            for i, (group_label, color) in enumerate(zip(self.group_names, colors)):
                time_points = self.target_time_points
                norm_values = norm_diff_matrix[i]
                smoothed = gaussian_filter1d(norm_values, sigma=2)
                ax.plot(time_points, smoothed, label=group_label if idx == 0 else "",
                        color=color, linewidth=2.5, alpha=0.9)

            display_title = title_mapping.get(class_pair, f"{class_pair[0]} vs {class_pair[1]}")
            ax.set_title(display_title, fontsize=18, fontweight='bold', fontfamily='Times New Roman', pad=15)
            ax.set_xlabel('Landing Phase (%)', fontsize=18, fontfamily='Times New Roman')
            ax.set_ylabel("Normalized Difference (Cohen's d)", fontsize=18, fontfamily='Times New Roman')

            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            effect_thresholds = [
                (0.2, 'gray', '--', 0.5),
                (0.5, 'gray', '--', 0.7),
                (0.8, 'gray', '--', 0.9)
            ]

            for threshold, color, linestyle, alpha in effect_thresholds:
                ax.axhline(y=threshold, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)
                ax.axhline(y=-threshold, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)

            ax.set_xlim(0, 100)
            all_norm_values = norm_diff_matrix.flatten()
            y_max = max(np.abs(all_norm_values).max() * 1.2, 1.0)
            ax.set_ylim(-y_max, y_max)
            ax.grid(True, alpha=0.3, linestyle='--')

            ax.annotate(label, xy=(-0.05, 1.02), xycoords='axes fraction',
                        fontsize=16, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        axes_flat[0].legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.25, wspace=0.25)

        plt.show()

        print("\nSaving figures...")
        filename = "combined_normalized_difference"
        save_figure(fig, filename)
        plt.close()

        print(f"\nCombined normalized difference plot saved")

    def generate_statistical_summary(self, analysis_result):
        class_pair = analysis_result['class_pair']
        norm_diff_matrix = analysis_result['norm_diff_matrix']
        abs_diff_matrix = analysis_result['abs_diff_matrix']

        print(f"\n{'=' * 70}")
        print(f"Statistical Summary for {class_pair[0]} vs {class_pair[1]}")
        print('=' * 70)

        summary_data = []

        for i, group_label in enumerate(self.group_names):
            norm_values = norm_diff_matrix[i]
            abs_values = abs_diff_matrix[i]

            mean_effect = np.mean(norm_values)
            max_effect = np.max(np.abs(norm_values))
            std_effect = np.std(norm_values)
            median_effect = np.median(norm_values)
            mean_abs = np.mean(abs_values)
            max_abs = np.max(abs_values)

            large_effect_points = np.sum(np.abs(norm_values) >= 0.8)
            medium_effect_points = np.sum((np.abs(norm_values) >= 0.5) & (np.abs(norm_values) < 0.8))
            small_effect_points = np.sum((np.abs(norm_values) >= 0.2) & (np.abs(norm_values) < 0.5))
            negligible_points = np.sum(np.abs(norm_values) < 0.2)

            max_idx = np.argmax(np.abs(norm_values))
            max_time = self.target_time_points[max_idx]

            max_abs_idx = np.argmax(abs_values)
            max_abs_time = self.target_time_points[max_abs_idx]

            summary_data.append([
                group_label,
                f"{mean_effect:.3f}",
                f"{max_effect:.3f}",
                f"{std_effect:.3f}",
                f"{median_effect:.3f}",
                f"{max_time:.1f}%",
                f"{mean_abs:.3f}",
                f"{max_abs:.3f}",
                f"{max_abs_time:.1f}%",
                f"{large_effect_points}",
                f"{medium_effect_points}",
                f"{small_effect_points}",
                f"{negligible_points}"
            ])

        summary_df = pd.DataFrame(summary_data, columns=[
            'Feature Group',
            'Mean Cohen\'s d', 'Max |Cohen\'s d|', 'Std Cohen\'s d', 'Median Cohen\'s d', 'Peak Effect Time',
            'Mean Abs Diff', 'Max Abs Diff', 'Peak Abs Diff Time',
            'Large Effect (|d|≥0.8)', 'Medium Effect (0.5≤|d|<0.8)',
            'Small Effect (0.2≤|d|<0.5)', 'Negligible (|d|<0.2)'
        ])

        print("\nNormalized Difference (Cohen's d) Statistics:")
        print(summary_df.to_string(index=False))

        filename = os.path.join(output_root, f"feature_analysis_{class_pair[0]}_vs_{class_pair[1]}.csv")
        summary_df.to_csv(filename, index=False)
        print(f"\nDetailed statistics saved to: {filename}")

        return summary_df

    def main_analysis(self):
        print("=" * 60)
        print("Biomechanical Feature Time Analysis")
        print("Resampling from 201 to 101 time points (0-100%)")
        print(f"Output directory: {output_root}")
        print("=" * 60)

        X, y, _ = self.load_and_prepare_data()

        comparisons = [
            ('40', '80'),
            ('F40', 'F80'),
            ('40', 'F40'),
            ('80', 'F80')
        ]

        all_results = {}

        for class_pair in comparisons:
            try:
                analysis_result = self.analyze_pair_comparison(class_pair, X, y)

                self.all_analysis_results[class_pair] = analysis_result

                print(f"\nGenerating visualizations for {class_pair[0]} vs {class_pair[1]}...")

                fig1 = self.plot_between_class_difference_heatmap(analysis_result)

                fig2 = self.plot_time_importance_for_each_feature_group(analysis_result)

                fig3 = self.plot_normalized_difference_comparison(analysis_result)

                summary_df = self.generate_statistical_summary(analysis_result)

                all_results[class_pair] = {
                    'analysis_result': analysis_result,
                    'summary_df': summary_df
                }

                print(f"\nAll figures saved for {class_pair[0]} vs {class_pair[1]}")

            except Exception as e:
                print(f"\nError analyzing {class_pair[0]} vs {class_pair[1]}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 80)
        print("Plotting combined heatmap (simple拼接)...")
        self.plot_combined_heatmap_simple()

        print("\n" + "=" * 80)
        print("Plotting combined normalized difference comparison...")
        self.plot_combined_normalized_difference()

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

        print(f"\nAll results saved to: {output_root}/")
        print("\nCombined figures:")
        print("   1. combined_heatmap_simple.png/svg/pdf - Combined Heatmap with (a)-(d) labels")
        print("   2. combined_normalized_difference.png/svg/pdf - Combined Normalized Difference with (a)-(d) labels")

        return all_results


if __name__ == "__main__":
    analyzer = BiomechanicsFeatureAnalyzer()
    results = analyzer.main_analysis()