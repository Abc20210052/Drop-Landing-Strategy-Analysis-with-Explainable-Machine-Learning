import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

output_root = 'SHAP_CrossFold_Interaction_Results'
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"Created main output directory: {output_root}")

file_paths = [
    r'F:\DL_40.csv',
    r'F:\DL_80.csv',
    r'F:\DL_P40.csv',
    r'F:\DL_P80.csv'
]

class_labels = ['40', '80', 'F40', 'F80']

BIOMECH_VARIABLES = {
    'Hip_Angle': {'start': 0, 'end': 201, 'color': '#E41A1C', 'label': 'Hip Angle'},
    'Hip_Moment': {'start': 201, 'end': 402, 'color': '#377EB8', 'label': 'Hip Moment'},
    'Knee_Angle': {'start': 402, 'end': 603, 'color': '#4DAF4A', 'label': 'Knee Angle'},
    'Knee_Moment': {'start': 603, 'end': 804, 'color': '#FF7F00', 'label': 'Knee Moment'},
    'Ankle_Angle': {'start': 804, 'end': 1005, 'color': '#984EA3', 'label': 'Ankle Angle'},
    'Ankle_Moment': {'start': 1005, 'end': 1206, 'color': '#A65628', 'label': 'Ankle Moment'},
    'Vertical_GRF': {'start': 1206, 'end': 1407, 'color': '#F781BF', 'label': 'Vertical GRF'}
}

BAR_COLORS = ['#2C6FB7', '#2D8F7A', '#D96C4A', '#A67B4A', '#7B4D8C', '#C95A7A', '#4A9FB5']
BAR_COLORS_ALTERNATIVE = ['#3A6FA8', '#4A9A88', '#D97A5A', '#B88A5A', '#8A5AA8', '#D06A8A', '#5A8FA8']

IMPORTANT_INTERACTIONS = [
    ('Hip_Angle', 'Knee_Angle'),
    ('Hip_Angle', 'Knee_Moment'),
    ('Knee_Angle', 'Knee_Moment'),
    ('Knee_Angle', 'Vertical_GRF'),
    ('Ankle_Angle', 'Vertical_GRF'),
    ('Hip_Moment', 'Knee_Moment'),
    ('Knee_Moment', 'Ankle_Moment'),
    ('Hip_Angle', 'Ankle_Angle'),
    ('Knee_Angle', 'Ankle_Angle'),
    ('Vertical_GRF', 'Knee_Moment'),
]


def save_figure(fig, base_path, dpi=300):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved PNG: {png_path}")
    svg_path = f"{base_path}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  Saved SVG: {svg_path}")
    pdf_path = f"{base_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved PDF: {pdf_path}")


class BiomechanicsDataProcessor:
    def __init__(self, file_paths, class_labels, normalize=True):
        self.file_paths = file_paths
        self.class_labels = class_labels
        self.normalize = normalize
        self.X = None
        self.X_normalized = None
        self.y = None
        self.subject_ids = None
        self.feature_names = None
        self.scaler = None

    def load_data(self):
        all_data = []
        all_labels = []
        all_subject_ids = []

        print("=" * 80)
        print("Starting data loading")
        print("=" * 80)

        for class_idx, file_path in enumerate(self.file_paths):
            print(f"\nProcessing class {self.class_labels[class_idx]}: {file_path}")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None)
                print(f"  Original data shape: {df.shape}")

                if df.shape == (61, 1408):
                    print(f"  Detected (61, 1408) shape, possibly containing header or index")
                    print(f"  Removing first row (header) and first column (index)")
                    df = df.iloc[1:, 1:]
                    print(f"  Processed shape: {df.shape}")

                if df.shape != (60, 1407):
                    print(f"  Warning: Expected shape (60, 1407), actual shape {df.shape}")
                    if df.shape[0] > 60:
                        df = df.iloc[:60, :]
                    if df.shape[1] > 1407:
                        df = df.iloc[:, :1407]
                    print(f"  Corrected shape: {df.shape}")

                try:
                    df = df.astype(np.float32)
                    print(f"  Data type converted to float32")
                except Exception as e:
                    print(f"  Data conversion error: {e}, attempting to fix...")
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.fillna(df.mean())

                for sample_idx in range(len(df)):
                    subject_id = self._create_subject_id(class_idx, sample_idx)
                    all_data.append(df.iloc[sample_idx].values)
                    all_labels.append(class_idx)
                    all_subject_ids.append(subject_id)

            else:
                print(f"  File does not exist, creating mock data...")
                self._create_mock_data(class_idx, all_data, all_labels, all_subject_ids)

        self.X = np.array(all_data, dtype=np.float32)
        self.y = np.array(all_labels)
        self.subject_ids = np.array(all_subject_ids)

        if self.normalize:
            print("\nPerforming data standardization...")
            self.scaler = StandardScaler()
            self.X_normalized = self.scaler.fit_transform(self.X)
            print(f"  Standardization complete, data range:")
            print(f"    Mean: {np.mean(self.X_normalized):.4f}")
            print(f"    Std: {np.std(self.X_normalized):.4f}")
            print(f"    Min: {np.min(self.X_normalized):.4f}")
            print(f"    Max: {np.max(self.X_normalized):.4f}")
        else:
            self.X_normalized = self.X.copy()
            print("\nSkipping data standardization")

        self.feature_names = self._create_feature_names()
        self._print_data_statistics()

        return self.X_normalized, self.y, self.subject_ids, self.feature_names

    def _create_subject_id(self, class_idx, sample_idx):
        subject_num = sample_idx // 3 + 1
        rep_num = sample_idx % 3 + 1
        return f"C{class_idx}_S{subject_num:02d}_R{rep_num}"

    def _create_mock_data(self, class_idx, all_data, all_labels, all_subject_ids):
        np.random.seed(42 + class_idx)
        mock_data = np.random.randn(60, 1407).astype(np.float32)
        mock_data += class_idx * 0.5
        for sample_idx in range(60):
            subject_id = self._create_subject_id(class_idx, sample_idx)
            all_data.append(mock_data[sample_idx])
            all_labels.append(class_idx)
            all_subject_ids.append(subject_id)

    def _create_feature_names(self):
        feature_names = []
        time_points = np.linspace(0, 100, 201)
        for var_name, var_info in BIOMECH_VARIABLES.items():
            start_idx = var_info['start']
            end_idx = var_info['end']
            for i in range(start_idx, end_idx):
                time_idx = i - start_idx
                time_point = time_points[time_idx]
                feature_names.append(f"{var_info['label']}_T{time_point:.1f}%")
        return feature_names

    def _print_data_statistics(self):
        print("\n" + "=" * 80)
        print("Data Statistics")
        print("=" * 80)
        print(f"Total samples: {len(self.X)}")
        print(f"Feature dimension: {self.X.shape[1]}")
        print(f"Data type: {self.X.dtype}")

        print(f"\nClass distribution:")
        for class_idx, label in enumerate(self.class_labels):
            count = np.sum(self.y == class_idx)
            print(f"  {label}: {count} samples")

        unique_subjects = np.unique(self.subject_ids)
        print(f"\nSubject statistics:")
        print(f"  Total subjects: {len(unique_subjects)}")

        print(f"\nOriginal data quality check:")
        print(f"  NaN count: {np.isnan(self.X).sum()}")
        print(f"  Inf count: {np.isinf(self.X).sum()}")
        print(f"  Data mean: {np.mean(self.X):.4f}")
        print(f"  Data std: {np.std(self.X):.4f}")
        print(f"  Data range: {np.min(self.X):.4f} to {np.max(self.X):.4f}")

        if self.normalize:
            print(f"\nNormalized data quality check:")
            print(f"  NaN count: {np.isnan(self.X_normalized).sum()}")
            print(f"  Inf count: {np.isinf(self.X_normalized).sum()}")
            print(f"  Data mean: {np.mean(self.X_normalized):.4f}")
            print(f"  Data std: {np.std(self.X_normalized):.4f}")
            print(f"  Data range: {np.min(self.X_normalized):.4f} to {np.max(self.X_normalized):.4f}")

        return True


class FeatureInteractionAnalyzer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.interaction_results = {}

    def analyze_interactions(self, model, X_test, shap_values, pair_name, output_dir):
        print("\n" + "=" * 60)
        print("Starting feature interaction analysis")
        print("=" * 60)

        interaction_dir = os.path.join(output_dir, "interaction_analysis")
        os.makedirs(interaction_dir, exist_ok=True)

        print("\n1. Computing variable-level aggregated features...")
        var_data = self._aggregate_variables(X_test)
        var_shap = self._aggregate_shap(shap_values)

        print("\n2. Computing variable interaction matrix...")
        interaction_matrix = self._compute_interaction_matrix(var_data, var_shap)

        self.interaction_results[pair_name] = {
            'matrix': interaction_matrix,
            'var_names': list(BIOMECH_VARIABLES.keys()),
            'var_labels': [BIOMECH_VARIABLES[var]['label'] for var in BIOMECH_VARIABLES.keys()]
        }

        self._plot_interaction_matrix(interaction_matrix, pair_name, interaction_dir)

        print("\n3. Analyzing important interaction pairs...")
        important_pairs = self._analyze_important_pairs(interaction_matrix, pair_name, interaction_dir)

        print("\n4. Plotting SHAP time curves...")
        self._plot_shap_curves(shap_values, pair_name, interaction_dir)

        print(f"\nInteraction analysis complete, results saved in: {interaction_dir}")
        return interaction_dir, important_pairs

    def _aggregate_variables(self, X_test):
        var_data = {}
        for var_name, var_info in BIOMECH_VARIABLES.items():
            start, end = var_info['start'], var_info['end']
            data = X_test[:, start:end]
            var_data[var_name] = {
                'mean': np.mean(data, axis=1),
                'max': np.max(data, axis=1),
                'min': np.min(data, axis=1),
                'std': np.std(data, axis=1)
            }
        return var_data

    def _aggregate_shap(self, shap_values):
        var_shap = {}
        for var_name, var_info in BIOMECH_VARIABLES.items():
            start, end = var_info['start'], var_info['end']
            shap = shap_values[:, start:end]
            var_shap[var_name] = {
                'mean_abs': np.mean(np.abs(shap), axis=1),
                'total_abs': np.sum(np.abs(shap), axis=1)
            }
        return var_shap

    def _compute_interaction_matrix(self, var_data, var_shap):
        var_names = list(BIOMECH_VARIABLES.keys())
        n = len(var_names)
        matrix = np.zeros((n, n))

        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i < j:
                    corr1 = np.corrcoef(var_data[var1]['mean'], var_data[var2]['mean'])[0, 1]
                    corr2 = np.corrcoef(var_shap[var1]['total_abs'], var_shap[var2]['total_abs'])[0, 1]

                    if np.isnan(corr1): corr1 = 0
                    if np.isnan(corr2): corr2 = 0

                    strength = (abs(corr1) + abs(corr2)) / 2
                    matrix[i, j] = strength
                    matrix[j, i] = strength
                elif i == j:
                    matrix[i, j] = 1.0

        return matrix

    def _plot_interaction_matrix(self, matrix, pair_name, output_dir):
        var_labels = [BIOMECH_VARIABLES[var]['label'] for var in BIOMECH_VARIABLES.keys()]

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Interaction Strength')

        ax.set_xticks(range(len(var_labels)))
        ax.set_yticks(range(len(var_labels)))
        ax.set_xticklabels(var_labels, rotation=45, ha='right', fontfamily='Times New Roman', fontsize=19)
        ax.set_yticklabels(var_labels, fontfamily='Times New Roman', fontsize=19)

        for i in range(len(var_labels)):
            for j in range(len(var_labels)):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                        color='black', fontfamily='Times New Roman', fontsize=15)

        ax.set_xlabel('Biomechanical Variables', fontfamily='Times New Roman', fontsize=22)
        ax.set_ylabel('Biomechanical Variables', fontfamily='Times New Roman', fontsize=22)

        plt.tight_layout()

        save_figure(fig, os.path.join(output_dir, 'interaction_matrix'))
        plt.close()

        df = pd.DataFrame(matrix, index=var_labels, columns=var_labels)
        df.to_csv(os.path.join(output_dir, 'interaction_matrix.csv'), encoding='utf-8-sig')

    def _analyze_important_pairs(self, matrix, pair_name, output_dir):
        var_names = list(BIOMECH_VARIABLES.keys())
        var_labels = [BIOMECH_VARIABLES[var]['label'] for var in var_names]

        results = []
        for var1, var2 in IMPORTANT_INTERACTIONS:
            if var1 in var_names and var2 in var_names:
                i, j = var_names.index(var1), var_names.index(var2)
                strength = matrix[i, j]
                results.append({
                    'Variable1': var1,
                    'Variable2': var2,
                    'Label1': BIOMECH_VARIABLES[var1]['label'],
                    'Label2': BIOMECH_VARIABLES[var2]['label'],
                    'Interaction_Strength': strength
                })

        df = pd.DataFrame(results)
        df = df.sort_values('Interaction_Strength', ascending=False)
        df.to_csv(os.path.join(output_dir, 'important_interactions.csv'), index=False, encoding='utf-8-sig')

        fig, ax = plt.subplots(figsize=(12, 8))
        top10 = df.head(10)

        y_pos = range(len(top10))
        labels = [f"{row['Label1']}\nvs\n{row['Label2']}" for _, row in top10.iterrows()]

        bars = ax.barh(y_pos, top10['Interaction_Strength'].values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontfamily='Times New Roman', fontsize=19)
        ax.set_xlabel('Interaction Strength', fontfamily='Times New Roman', fontsize=19)
        ax.set_title(f'Top 10 Important Interactions - {pair_name}',
                     fontfamily='Times New Roman', fontsize=17, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, top10['Interaction_Strength'].values)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', ha='left', fontfamily='Times New Roman', fontsize=16)

        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'important_interactions_ranking'))
        plt.close()

        return results

    def _plot_shap_curves(self, shap_values, pair_name, output_dir):
        time_points = np.linspace(0, 100, 201)

        for var_name, var_info in BIOMECH_VARIABLES.items():
            start, end = var_info['start'], var_info['end']
            var_shap = shap_values[:, start:end]

            mean_shap = np.mean(np.abs(var_shap), axis=0)
            std_shap = np.std(np.abs(var_shap), axis=0)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_points, mean_shap, 'b-', linewidth=2, label='Mean |SHAP|')
            ax.fill_between(time_points, mean_shap - std_shap, mean_shap + std_shap, alpha=0.3, color='blue')

            ax.set_xlabel('Landing Phase (%)', fontfamily='Times New Roman', fontsize=17)
            ax.set_ylabel('Mean |SHAP| Value', fontfamily='Times New Roman', fontsize=17)
            ax.set_title(f'{var_info["label"]} - {pair_name}',
                         fontfamily='Times New Roman', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(prop={'family': 'Times New Roman'}, fontsize=14)
            ax.set_xlim(0, 100)
            ax.tick_params(labelsize=14)

            plt.tight_layout()
            save_figure(fig, os.path.join(output_dir, f'shap_curve_{var_name}'))
            plt.close()

    def plot_interaction_matrices_combined(self):
        print("\n" + "=" * 80)
        print("Plotting combined interaction matrices")
        print("=" * 80)

        if not self.interaction_results:
            print("No interaction matrix data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        axes_flat = axes.flatten()

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            '40_vs_80': 'Task 1 vs Task 2',
            'F40_vs_F80': 'Task 3 vs Task 4',
            '40_vs_F40': 'Task 1 vs Task 3',
            '80_vs_F80': 'Task 1 vs Task 4'
        }

        pair_names = list(self.interaction_results.keys())

        for idx, (ax, pair_name, label) in enumerate(zip(axes_flat, pair_names, subplot_labels)):
            data = self.interaction_results[pair_name]
            matrix = data['matrix']
            var_labels = data['var_labels']

            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(range(len(var_labels)))
            ax.set_yticks(range(len(var_labels)))
            ax.set_xticklabels(var_labels, rotation=45, ha='right', fontfamily='Times New Roman', fontsize=13)
            ax.set_yticklabels(var_labels, fontfamily='Times New Roman', fontsize=13)

            for i in range(len(var_labels)):
                for j in range(len(var_labels)):
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                            color='black', fontfamily='Times New Roman', fontsize=11)

            display_title = title_mapping.get(pair_name, pair_name)
            ax.set_title(f'{display_title}', fontfamily='Times New Roman', fontsize=17, fontweight='bold')

            ax.set_xlabel('Biomechanical Variables', fontfamily='Times New Roman', fontsize=15)
            ax.set_ylabel('Biomechanical Variables', fontfamily='Times New Roman', fontsize=15)

            ax.annotate(label, xy=(-0.05, 1.05), xycoords='axes fraction',
                        fontsize=21, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Interaction Strength', fontfamily='Times New Roman', fontsize=15)
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.9)

        output_path = os.path.join(output_root, 'interaction_matrices_combined')
        save_figure(fig, output_path)
        plt.show()

        print(f"\nCombined interaction matrices plot saved: {output_path}.png/.svg")

    def _get_pair_titles(self, pair_names):
        title_mapping = {
            '40_vs_80': '40 vs 80',
            'F40_vs_F80': 'F40 vs F80',
            '40_vs_F40': '40 vs F40',
            '80_vs_F80': '80 vs F80'
        }
        return [title_mapping.get(name, name) for name in pair_names]


class SHAPCrossFoldAnalyzer:
    def __init__(self, class_labels, feature_names, n_folds=5):
        self.class_labels = class_labels
        self.feature_names = feature_names
        self.n_folds = n_folds
        self.results = {}
        self.all_shap_data = {}
        self.interaction_analyzer = FeatureInteractionAnalyzer(feature_names)

    def analyze_binary_pair_crossfold(self, class1_idx, class2_idx, X, y, subject_ids):
        pair_name = f"{self.class_labels[class1_idx]}_vs_{self.class_labels[class2_idx]}"
        pair_title = self._get_pair_title(class1_idx, class2_idx)

        print(f"\n{'=' * 80}")
        print(f"Cross-Fold SHAP Analysis: {pair_title}")
        print(f"Number of cross-validation folds: {self.n_folds}")
        print(f"{'=' * 80}")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, f"CrossFold_{pair_name}_{current_time}")
        os.makedirs(output_dir, exist_ok=True)

        binary_data = self._prepare_binary_data(class1_idx, class2_idx, X, y, subject_ids)
        X_binary, y_binary, subject_ids_binary = binary_data

        fold_shap_values = []
        fold_accuracies = []
        fold_var_importance = []
        fold_models = []
        fold_feature_importance = []

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X_binary, y_binary)):
            print(f"\n  Fold {fold_idx + 1}/{self.n_folds}")

            X_train = X_binary[train_index]
            X_test = X_binary[test_index]
            y_train = y_binary[train_index]
            y_test = y_binary[test_index]

            train_scaler = StandardScaler()
            X_train_scaled = train_scaler.fit_transform(X_train)
            X_test_scaled = train_scaler.transform(X_test)

            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.01,
                num_leaves=15,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train_scaled, y_train)
            fold_models.append(model)

            accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            fold_accuracies.append(accuracy)
            print(f"    Accuracy: {accuracy:.4f}")

            explainer = shap.TreeExplainer(model, X_train_scaled)
            shap_values = explainer.shap_values(X_test_scaled)

            if isinstance(shap_values, list):
                shap_values_2d = shap_values[1]
            else:
                shap_values_2d = shap_values

            fold_shap_values.append(shap_values_2d)

            var_imp = self._calculate_variable_importance(shap_values_2d)
            fold_var_importance.append(var_imp)

            feature_imp = np.sum(np.abs(shap_values_2d), axis=0)
            fold_feature_importance.append(feature_imp)

        avg_shap_values = np.mean(fold_shap_values, axis=0)
        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)

        print(f"\nCross-validation results: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

        X_test_combined = None
        for fold_idx, (_, test_index) in enumerate(skf.split(X_binary, y_binary)):
            X_test_fold = X_binary[test_index]
            if X_test_combined is None:
                X_test_combined = X_test_fold
            else:
                X_test_combined = np.vstack([X_test_combined, X_test_fold])

        scaler_combined = StandardScaler()
        X_test_combined_scaled = scaler_combined.fit_transform(X_test_combined)

        np.save(os.path.join(output_dir, "shap_values.npy"), avg_shap_values)

        avg_var_importance = self._average_variable_importance(fold_var_importance)
        self._save_variable_importance(avg_var_importance, pair_name, output_dir)

        final_model = fold_models[-1]
        interaction_dir, important_pairs = self.interaction_analyzer.analyze_interactions(
            final_model, X_test_combined_scaled, avg_shap_values, pair_title, output_dir
        )

        self.all_shap_data[pair_name] = {
            'shap_values': avg_shap_values,
            'X_test': X_test_combined_scaled,
            'title': pair_title,
            'accuracy': avg_accuracy,
            'accuracy_std': std_accuracy,
            'class1': self.class_labels[class1_idx],
            'class2': self.class_labels[class2_idx],
            'var_importance': avg_var_importance,
            'important_pairs': important_pairs,
            'all_accuracies': fold_accuracies
        }

        self._save_report(pair_name, avg_accuracy, std_accuracy, avg_var_importance,
                          important_pairs, fold_accuracies, output_dir, interaction_dir)

        self.results[pair_name] = {
            'accuracy': avg_accuracy,
            'accuracy_std': std_accuracy,
            'all_accuracies': fold_accuracies,
            'output_dir': output_dir,
            'interaction_dir': interaction_dir
        }

        return self.results[pair_name]

    def _get_pair_title(self, class1_idx, class2_idx):
        pairs = {
            (0, 1): "Task 1 vs Task 2",
            (2, 3): "Task 3 vs Task 4",
            (0, 2): "Task 1 vs Task 3",
            (1, 3): "Task 1 vs Task 4"
        }
        return pairs.get((class1_idx, class2_idx),
                         f"{self.class_labels[class1_idx]} vs {self.class_labels[class2_idx]}")

    def _prepare_binary_data(self, class1_idx, class2_idx, X, y, subject_ids):
        mask = (y == class1_idx) | (y == class2_idx)
        X_binary = X[mask]
        y_binary = y[mask]
        subject_ids_binary = subject_ids[mask]
        y_binary = np.where(y_binary == class1_idx, 0, 1)

        print(f"Binary data:")
        print(f"  Samples: {len(X_binary)}")
        print(f"  Class distribution: {self.class_labels[class1_idx]}: {np.sum(y_binary == 0)}, "
              f"{self.class_labels[class2_idx]}: {np.sum(y_binary == 1)}")

        return X_binary, y_binary, subject_ids_binary

    def _calculate_variable_importance(self, shap_values):
        variable_importance = []
        shap_dim = shap_values.shape[1]

        for var_name, var_info in BIOMECH_VARIABLES.items():
            start_idx = var_info['start']
            end_idx = var_info['end']

            if start_idx >= shap_dim:
                continue
            if end_idx > shap_dim:
                end_idx = shap_dim

            var_shap = shap_values[:, start_idx:end_idx]
            total_importance = np.sum(np.abs(var_shap))
            mean_importance = np.mean(np.abs(var_shap))

            if var_shap.shape[1] > 0:
                time_importance = np.mean(np.abs(var_shap), axis=0)
                max_time_idx = np.argmax(time_importance)
                max_time_percent = (max_time_idx / max(var_shap.shape[1] - 1, 1)) * 100
            else:
                max_time_percent = 0

            variable_importance.append({
                'variable': var_info['label'],
                'code': var_name,
                'total_importance': total_importance,
                'mean_importance': mean_importance,
                'max_time_point': f"{max_time_percent:.1f}%",
                'color': var_info['color']
            })

        variable_importance.sort(key=lambda x: x['total_importance'], reverse=True)
        return variable_importance

    def _average_variable_importance(self, fold_var_importance):
        var_names = [item['variable'] for item in fold_var_importance[0]]
        avg_importance = []

        for i, var_name in enumerate(var_names):
            total_imp_list = [fold[i]['total_importance'] for fold in fold_var_importance]
            mean_imp_list = [fold[i]['mean_importance'] for fold in fold_var_importance]
            max_time = fold_var_importance[0][i]['max_time_point']

            avg_importance.append({
                'variable': var_name,
                'total_importance': np.mean(total_imp_list),
                'mean_importance': np.mean(mean_imp_list),
                'max_time_point': max_time,
                'color': BIOMECH_VARIABLES[list(BIOMECH_VARIABLES.keys())[i]]['color']
            })

        avg_importance.sort(key=lambda x: x['total_importance'], reverse=True)
        return avg_importance

    def _save_variable_importance(self, var_importance, pair_name, output_dir):
        df = pd.DataFrame(var_importance)
        df.to_csv(os.path.join(output_dir, 'variable_importance.csv'), index=False, encoding='utf-8-sig')

    def _save_report(self, pair_name, avg_accuracy, std_accuracy, var_importance,
                     important_pairs, all_accuracies, output_dir, interaction_dir):
        report_path = os.path.join(output_dir, "analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Cross-Fold SHAP Analysis Report - {pair_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: LightGBM (n_estimators=200, learning_rate=0.01, num_leaves=15, max_depth=5)\n")
            f.write(f"Number of cross-validation folds: {self.n_folds}\n")
            f.write(f"Mean accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
            f.write(f"Per-fold accuracy: {', '.join([f'{acc:.4f}' for acc in all_accuracies])}\n\n")

            f.write("Variable Importance Ranking (Total SHAP Importance):\n")
            f.write("-" * 30 + "\n")
            for i, var in enumerate(var_importance, 1):
                f.write(f"{i}. {var['variable']}: {var['total_importance']:.2f} "
                        f"(peak at {var['max_time_point']})\n")

            f.write("\n\nImportant Interaction Pairs Ranking:\n")
            f.write("-" * 30 + "\n")
            if important_pairs:
                for i, pair in enumerate(important_pairs[:5], 1):
                    f.write(f"{i}. {pair['Label1']} vs {pair['Label2']}: {pair['Interaction_Strength']:.3f}\n")

            f.write(f"\nFeature interaction analysis results saved in: {interaction_dir}\n")

        print(f"\nAnalysis report saved: {report_path}")

    def plot_combined_results(self):
        print("\n" + "=" * 80)
        print("Plotting combined results")
        print("=" * 80)

        if not self.all_shap_data:
            print("No data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        axes_flat = axes.flatten()

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            '40_vs_80': 'Task 1 vs Task 2',
            'F40_vs_F80': 'Task 3 vs Task 4',
            '40_vs_F40': 'Task 1 vs Task 3',
            '80_vs_F80': 'Task 1 vs Task 4'
        }

        for idx, (ax, pair_name, label) in enumerate(zip(axes_flat, list(self.all_shap_data.keys()), subplot_labels)):
            data = self.all_shap_data[pair_name]
            var_importance = data['var_importance']

            var_names = [item['variable'] for item in var_importance]
            total_imp = [item['total_importance'] for item in var_importance]
            colors = [item['color'] for item in var_importance]

            y_pos = np.arange(len(var_names))
            bars = ax.barh(y_pos, total_imp, color=colors, alpha=0.85,
                           edgecolor='#2C3E50', linewidth=1.2)

            for i, (bar, val) in enumerate(zip(bars, total_imp)):
                ax.text(val + max(total_imp) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', va='center', ha='left',
                        fontfamily='Times New Roman', fontsize=14)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(var_names, fontfamily='Times New Roman', fontsize=15)
            ax.set_xlabel('Total SHAP Importance', fontfamily='Times New Roman', fontsize=17)

            display_title = title_mapping.get(pair_name, data['title'])
            ax.set_title(f'{display_title}',
                         fontfamily='Times New Roman', fontsize=17, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            ax.tick_params(axis='x', labelsize=15)

            ax.annotate(label, xy=(-0.05, 1.05), xycoords='axes fraction',
                        fontsize=21, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        output_path = os.path.join(output_root, 'shap_combined_results')
        save_figure(fig, output_path)
        plt.show()

        print(f"\nCombined results plot saved: {output_path}.png/.svg")

    def plot_shap_time_curves_combined(self):
        print("\n" + "=" * 80)
        print("Plotting combined SHAP time curves")
        print("=" * 80)

        time_points = np.linspace(0, 100, 201)
        pair_names = list(self.all_shap_data.keys())
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            '40_vs_80': 'Task 1 vs Task 2',
            'F40_vs_F80': 'Task 3 vs Task 4',
            '40_vs_F40': 'Task 1 vs Task 3',
            '80_vs_F80': 'Task 1 vs Task 4'
        }

        for var_name, var_info in BIOMECH_VARIABLES.items():
            fig, axes = plt.subplots(2, 2, figsize=(22, 18))
            axes_flat = axes.flatten()

            for idx, (ax, pair_name, label) in enumerate(zip(axes_flat, pair_names, subplot_labels)):
                data = self.all_shap_data[pair_name]
                shap_values = data['shap_values']

                start, end = var_info['start'], var_info['end']
                var_shap = shap_values[:, start:end]

                mean_shap = np.mean(np.abs(var_shap), axis=0)
                std_shap = np.std(np.abs(var_shap), axis=0)

                ax.plot(time_points, mean_shap, 'b-', linewidth=2.5, label='Mean |SHAP|')
                ax.fill_between(time_points,
                                mean_shap - std_shap,
                                mean_shap + std_shap,
                                alpha=0.3, color='blue')

                ax.set_xlabel('Landing Phase (%)', fontfamily='Times New Roman', fontsize=15)
                ax.set_ylabel('Mean |SHAP| Value', fontfamily='Times New Roman', fontsize=15)

                display_title = title_mapping.get(pair_name, data['title'])
                ax.set_title(f'{display_title}', fontfamily='Times New Roman', fontsize=15, fontweight='bold')
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.set_xlim(0, 100)
                ax.legend(fontsize=13, prop={'family': 'Times New Roman'})
                ax.tick_params(labelsize=14)

                ax.annotate(label, xy=(-0.05, 1.05), xycoords='axes fraction',
                            fontsize=21, fontweight='bold', va='center', ha='left',
                            fontfamily='Times New Roman')

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            output_path = os.path.join(output_root, f'shap_curves_{var_name}_combined')
            save_figure(fig, output_path)
            plt.close()

            print(f"  {var_info['label']} time curves plot saved")

    def plot_peak_times_combined(self):
        print("\n" + "=" * 80)
        print("Plotting combined peak times")
        print("=" * 80)

        time_points = np.linspace(0, 100, 201)
        pair_names = list(self.all_shap_data.keys())
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            '40_vs_80': 'Task 1 vs Task 2',
            'F40_vs_F80': 'Task 3 vs Task 4',
            '40_vs_F40': 'Task 1 vs Task 3',
            '80_vs_F80': 'Task 1 vs Task 4'
        }

        peak_times = {}
        for var_name, var_info in BIOMECH_VARIABLES.items():
            peak_times[var_info['label']] = []
            for pair_name in pair_names:
                data = self.all_shap_data[pair_name]
                shap_values = data['shap_values']

                start, end = var_info['start'], var_info['end']
                var_shap = shap_values[:, start:end]
                mean_shap = np.mean(np.abs(var_shap), axis=0)
                peak_time = time_points[np.argmax(mean_shap)]
                peak_times[var_info['label']].append(peak_time)

        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        axes_flat = axes.flatten()

        for idx, (ax, pair_name, label) in enumerate(zip(axes_flat, pair_names, subplot_labels)):
            var_labels = []
            times = []
            colors = []

            for var_name, var_info in BIOMECH_VARIABLES.items():
                var_labels.append(var_info['label'])
                times.append(peak_times[var_info['label']][idx])
                colors.append(var_info['color'])

            sorted_idx = np.argsort(times)
            sorted_labels = [var_labels[i] for i in sorted_idx]
            sorted_times = [times[i] for i in sorted_idx]
            sorted_colors = [colors[i] for i in sorted_idx]

            bars = ax.barh(range(len(sorted_labels)), sorted_times,
                           color=sorted_colors, alpha=0.8)

            ax.set_yticks(range(len(sorted_labels)))
            ax.set_yticklabels(sorted_labels, fontfamily='Times New Roman', fontsize=15)
            ax.set_xlabel('Peak Time (% Landing Phase)', fontfamily='Times New Roman', fontsize=17)

            display_title = title_mapping.get(pair_name, self.all_shap_data[pair_name]['title'])
            ax.set_title(f'{display_title}',
                         fontfamily='Times New Roman', fontsize=17, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            ax.tick_params(axis='x', labelsize=15)

            for i, (bar, time) in enumerate(zip(bars, sorted_times)):
                ax.text(time + 1, bar.get_y() + bar.get_height() / 2,
                        f'{time:.1f}%', va='center', ha='left',
                        fontfamily='Times New Roman', fontsize=13)

            ax.annotate(label, xy=(-0.05, 1.05), xycoords='axes fraction',
                        fontsize=21, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        output_path = os.path.join(output_root, 'peak_times_combined')
        save_figure(fig, output_path)
        plt.show()

        print(f"\nCombined peak times plot saved: {output_path}.png/.svg")


def main():
    print("=" * 80)
    print("SHAP Cross-Validation Analysis - 5-Fold Cross-Validation (with Feature Interaction Analysis)")
    print("Model: LightGBM (n_estimators=200, learning_rate=0.01, num_leaves=15, max_depth=5)")
    print("Variable Importance: Total SHAP Importance")
    print("=" * 80)
    print(f"Output directory: {output_root}")
    print("=" * 80)

    print("\n1. Loading and standardizing data...")
    processor = BiomechanicsDataProcessor(file_paths, class_labels, normalize=True)
    X, y, subject_ids, feature_names = processor.load_data()

    class_pairs = [
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3)
    ]

    print("\n2. Starting cross-validation SHAP analysis (5-fold)...")
    crossfold_analyzer = SHAPCrossFoldAnalyzer(
        class_labels,
        feature_names,
        n_folds=5
    )

    all_results = {}
    for class1_idx, class2_idx in class_pairs:
        result = crossfold_analyzer.analyze_binary_pair_crossfold(
            class1_idx, class2_idx, X, y, subject_ids
        )
        pair_name = f"{class_labels[class1_idx]}_vs_{class_labels[class2_idx]}"
        all_results[pair_name] = result

    print("\n3. Plotting combined results...")

    crossfold_analyzer.plot_combined_results()

    crossfold_analyzer.plot_shap_time_curves_combined()

    crossfold_analyzer.plot_peak_times_combined()

    if hasattr(crossfold_analyzer.interaction_analyzer, 'plot_interaction_matrices_combined'):
        crossfold_analyzer.interaction_analyzer.plot_interaction_matrices_combined()
    else:
        print("Warning: Combined interaction matrices plot method not found")

    print("\n" + "=" * 80)
    print("Cross-Validation SHAP Analysis Complete!")
    print("=" * 80)

    print(f"\nAll results saved to: {output_root}/")
    print("   - Each comparison has its own CrossFold folder")
    print("   - All figures saved as PNG, SVG, and PDF formats")
    print("   - Includes per-fold accuracy, variable importance with CI, and stability analysis")
    print("   - Variable importance uses Total SHAP Importance")
    print("   - Feature interaction analysis saved in interaction_analysis/ subdirectory")
    print("   - Interaction matrix displays all values in black text")

    return all_results


if __name__ == "__main__":
    results = main()