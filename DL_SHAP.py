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

output_root = 'SHAP_CrossFold_Results'
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


def save_figure(fig, base_path, dpi=300):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=dpi)
    print(f"  ✓ Saved PNG: {png_path}")
    svg_path = f"{base_path}.svg"
    fig.savefig(svg_path, format='svg')
    print(f"  ✓ Saved SVG: {svg_path}")
    pdf_path = f"{base_path}.pdf"
    fig.savefig(pdf_path, format='pdf')
    print(f"  ✓ Saved PDF: {pdf_path}")


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


class SHAPCrossFoldAnalyzer:
    def __init__(self, class_labels, feature_names, n_folds=5):
        self.class_labels = class_labels
        self.feature_names = feature_names
        self.n_folds = n_folds
        self.crossfold_results = {}
        self.all_crossfold_data = {}

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
        fold_train_indices = []
        fold_test_indices = []

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X_binary, y_binary)):
            print(f"\n  Fold {fold_idx + 1}/{self.n_folds}")

            X_train = X_binary[train_index]
            X_test = X_binary[test_index]
            y_train = y_binary[train_index]
            y_test = y_binary[test_index]

            fold_train_indices.append(train_index)
            fold_test_indices.append(test_index)

            train_scaler = StandardScaler()
            X_train_scaled = train_scaler.fit_transform(X_train)
            X_test_scaled = train_scaler.transform(X_test)

            model = self._train_model(X_train_scaled, y_train)
            fold_models.append(model)

            accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            fold_accuracies.append(accuracy)
            print(f"    Accuracy: {accuracy:.4f}")

            if len(X_train_scaled) > 100:
                background = shap.sample(X_train_scaled, 100)
            else:
                background = X_train_scaled

            explainer = shap.TreeExplainer(model, background)
            shap_values = explainer.shap_values(X_test_scaled)

            if isinstance(shap_values, list):
                shap_values_2d = shap_values[1]
            else:
                shap_values_2d = shap_values

            fold_shap_values.append(shap_values_2d)

            var_imp = self._calculate_variable_importance(shap_values_2d)
            fold_var_importance.append(var_imp)

            feature_imp = np.abs(shap_values_2d).mean(axis=0)
            fold_feature_importance.append(feature_imp)

        crossfold_results = self._analyze_crossfold_results(
            fold_shap_values,
            fold_accuracies,
            fold_var_importance,
            fold_feature_importance,
            fold_models,
            fold_train_indices,
            fold_test_indices,
            pair_title,
            output_dir
        )

        self.all_crossfold_data[pair_name] = {
            'var_importance_stats': crossfold_results['var_importance_stats'],
            'title': pair_title,
            'accuracy_mean': crossfold_results['accuracy_stats']['mean'],
            'accuracy_std': crossfold_results['accuracy_stats']['std'],
            'accuracy_ci': (crossfold_results['accuracy_stats']['ci_lower'],
                            crossfold_results['accuracy_stats']['ci_upper']),
            'n_folds': self.n_folds
        }

        self.crossfold_results[pair_name] = crossfold_results

        return crossfold_results

    def _analyze_crossfold_results(self, fold_shap_values, fold_accuracies,
                                   fold_var_importance, fold_feature_importance,
                                   fold_models, fold_train_indices, fold_test_indices,
                                   pair_title, output_dir):
        print(f"\nAnalyzing cross-validation results...")

        accuracy_stats = {
            'mean': np.mean(fold_accuracies),
            'std': np.std(fold_accuracies),
            'ci_lower': np.percentile(fold_accuracies, 2.5),
            'ci_upper': np.percentile(fold_accuracies, 97.5),
            'min': np.min(fold_accuracies),
            'max': np.max(fold_accuracies),
            'all': fold_accuracies
        }

        print(f"\nAccuracy statistics (based on {self.n_folds}-fold cross-validation):")
        print(f"  Mean: {accuracy_stats['mean']:.4f} ± {accuracy_stats['std']:.4f}")
        print(f"  95% CI: [{accuracy_stats['ci_lower']:.4f}, {accuracy_stats['ci_upper']:.4f}]")

        var_importance_stats = self._calculate_var_importance_stats(fold_var_importance)
        shap_stability = self._calculate_shap_stability(fold_shap_values)

        self._plot_crossfold_results(
            fold_accuracies,
            fold_shap_values,
            var_importance_stats,
            shap_stability,
            fold_feature_importance,
            pair_title,
            output_dir
        )

        self._plot_variable_importance_with_ci(
            var_importance_stats,
            pair_title,
            output_dir
        )

        self._plot_fold_comparison(
            fold_var_importance,
            pair_title,
            output_dir
        )

        self._save_crossfold_results(
            accuracy_stats,
            var_importance_stats,
            shap_stability,
            pair_title,
            output_dir
        )

        return {
            'accuracy_stats': accuracy_stats,
            'var_importance_stats': var_importance_stats,
            'shap_stability': shap_stability,
            'shap_values': fold_shap_values,
            'models': fold_models,
            'output_dir': output_dir
        }

    def _calculate_shap_stability(self, fold_shap_values):
        n_features = fold_shap_values[0].shape[1]

        for shap_val in fold_shap_values:
            if shap_val.shape[1] != n_features:
                n_features = min(n_features, shap_val.shape[1])

        feature_stds = []

        for feature_idx in range(n_features):
            feature_values_list = []
            for shap_val in fold_shap_values:
                if shap_val.shape[1] > feature_idx:
                    feature_values_list.append(shap_val[:, feature_idx])

            if feature_values_list:
                all_values = np.concatenate(feature_values_list)
                feature_std = np.std(all_values)
                feature_stds.append(feature_std)
            else:
                feature_stds.append(0.0)

        return {
            'feature_stds': np.array(feature_stds),
            'overall_stability': 1 / (1 + np.mean(feature_stds)) if np.mean(feature_stds) > 0 else 1.0,
            'mean_std': np.mean(feature_stds)
        }

    def _calculate_var_importance_stats(self, fold_var_importance):
        var_names = [item['variable'] for item in fold_var_importance[0]]
        n_folds = len(fold_var_importance)

        importance_matrix = np.zeros((n_folds, len(var_names)))
        for i, var_imp in enumerate(fold_var_importance):
            for j, item in enumerate(var_imp):
                importance_matrix[i, j] = item['total_importance']

        stats = {
            'var_names': var_names,
            'mean': np.mean(importance_matrix, axis=0),
            'std': np.std(importance_matrix, axis=0),
            'ci_lower': np.percentile(importance_matrix, 2.5, axis=0),
            'ci_upper': np.percentile(importance_matrix, 97.5, axis=0),
            'importance_matrix': importance_matrix
        }

        return stats

    def _get_pair_title(self, class1_idx, class2_idx):
        pairs = {
            (0, 1): "40 vs 80 (Pre-fatigue)",
            (2, 3): "F40 vs F80 (Post-fatigue)",
            (0, 2): "40 vs F40 (Fatigue Effect)",
            (1, 3): "80 vs F80 (Fatigue Effect)"
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

    def _train_model(self, X_train, y_train):
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        return model

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

    def _plot_crossfold_results(self, fold_accuracies, fold_shap_values,
                                var_importance_stats, shap_stability, fold_feature_importance,
                                pair_title, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        ax1 = axes[0, 0]
        folds = range(1, len(fold_accuracies) + 1)
        bar_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(folds)))[::-1]
        ax1.bar(folds, fold_accuracies, color=bar_colors, alpha=0.8, edgecolor='#2C3E50', linewidth=1.2)
        ax1.axhline(np.mean(fold_accuracies), color='#C0392B', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}')
        ax1.axhline(np.mean(fold_accuracies) - np.std(fold_accuracies), color='#7F8C8D', linestyle=':', linewidth=1.5)
        ax1.axhline(np.mean(fold_accuracies) + np.std(fold_accuracies), color='#7F8C8D', linestyle=':', linewidth=1.5)
        ax1.set_xlabel('Fold', fontsize=14, fontfamily='Times New Roman')
        ax1.set_ylabel('Accuracy', fontsize=14, fontfamily='Times New Roman')
        ax1.set_title(f'Accuracy Across {self.n_folds} Folds',
                      fontsize=16, fontweight='bold', fontfamily='Times New Roman')
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.05])

        ax2 = axes[0, 1]
        var_names = var_importance_stats['var_names']
        mean_imp = var_importance_stats['mean']
        std_imp = var_importance_stats['std']

        y_pos = np.arange(len(var_names))
        bars = ax2.barh(y_pos, mean_imp, xerr=std_imp, color=BAR_COLORS[:len(var_names)],
                        alpha=0.85, edgecolor='#2C3E50', linewidth=1.2, capsize=6)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(var_names, fontsize=12, fontfamily='Times New Roman')
        ax2.set_xlabel('Mean Total SHAP Importance', fontsize=14, fontfamily='Times New Roman')
        ax2.set_ylabel('Biomechanical Feature', fontsize=14, fontfamily='Times New Roman')
        ax2.set_title(f'Variable Importance with Cross-Fold Error Bars',
                      fontsize=16, fontweight='bold', fontfamily='Times New Roman')
        ax2.grid(True, alpha=0.3, axis='x')

        ax3 = axes[1, 0]
        feature_stds = shap_stability['feature_stds']

        var_stability = []
        var_names_short = []
        stability_colors = []
        for var_name, var_info in BIOMECH_VARIABLES.items():
            start_idx = var_info['start']
            end_idx = min(var_info['end'], len(feature_stds))
            if start_idx < len(feature_stds):
                var_std = np.mean(feature_stds[start_idx:end_idx])
                var_stability.append(var_std)
                var_names_short.append(var_info['label'])
                stability_colors.append(var_info['color'])

        bars = ax3.barh(var_names_short, var_stability, color=stability_colors, alpha=0.7,
                        edgecolor='#2C3E50', linewidth=1)

        ax3.set_xlabel('SHAP Value Standard Deviation (across folds)',
                       fontsize=14, fontfamily='Times New Roman')
        ax3.set_ylabel('Biomechanical Feature', fontsize=14, fontfamily='Times New Roman')
        ax3.set_title(f'SHAP Stability Across {self.n_folds} Folds',
                      fontsize=16, fontweight='bold', fontfamily='Times New Roman')
        ax3.grid(True, alpha=0.3, axis='x')

        ax4 = axes[1, 1]
        last_shap = fold_shap_values[-1]
        feature_importance = np.abs(last_shap).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-20:]
        top_features = [self.feature_names[i].split('_T')[0] if '_T' in self.feature_names[i]
                        else self.feature_names[i] for i in top_indices]

        shap_data = last_shap[:, top_indices]
        shap.summary_plot(
            shap_data,
            np.zeros_like(shap_data),
            feature_names=top_features,
            max_display=20,
            plot_size=(6, 6),
            show=False
        )
        plt.title(f'SHAP Summary (Last Fold)', fontsize=16, fontweight='bold',
                  fontfamily='Times New Roman')

        plt.suptitle(f'Cross-Fold SHAP Analysis - {pair_title} ({self.n_folds}-fold CV)',
                     fontsize=18, fontweight='bold', fontfamily='Times New Roman', y=0.98)

        plt.tight_layout()

        base_path = os.path.join(output_dir, "crossfold_results")
        save_figure(fig, base_path)
        plt.close()

    def _plot_variable_importance_with_ci(self, var_importance_stats, pair_title, output_dir):
        fig, ax = plt.subplots(figsize=(12, 8))

        var_names = var_importance_stats['var_names']
        mean_imp = var_importance_stats['mean']
        ci_lower = var_importance_stats['ci_lower']
        ci_upper = var_importance_stats['ci_upper']

        y_pos = np.arange(len(var_names))

        bars = ax.barh(y_pos, mean_imp, color=BAR_COLORS_ALTERNATIVE[:len(var_names)],
                       alpha=0.85, edgecolor='#2C3E50', linewidth=1.2)

        for i, (bar, lower, upper) in enumerate(zip(bars, ci_lower, ci_upper)):
            ax.plot([lower, upper], [bar.get_y() + bar.get_height() / 2,
                                     bar.get_y() + bar.get_height() / 2],
                    color='#2C3E50', linewidth=2.5, marker='|', markersize=10)

            ax.text(mean_imp[i] + max(mean_imp) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{mean_imp[i]:.2f}', va='center', ha='left',
                    fontsize=11, fontfamily='Times New Roman', color='#2C3E50')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names, fontsize=13, fontfamily='Times New Roman')
        ax.set_xlabel('Total SHAP Importance', fontsize=15, fontfamily='Times New Roman')
        ax.set_ylabel('Biomechanical Feature', fontsize=15, fontfamily='Times New Roman')
        ax.set_title(f'Variable Importance with 95% Confidence Intervals ({self.n_folds}-fold CV)',
                     fontsize=17, fontweight='bold', fontfamily='Times New Roman')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

        plt.tight_layout()

        base_path = os.path.join(output_dir, "variable_importance_with_ci")
        save_figure(fig, base_path)
        plt.close()

    def _plot_fold_comparison(self, fold_var_importance, pair_title, output_dir):
        fig, ax = plt.subplots(figsize=(14, 8))

        var_names = [item['variable'] for item in fold_var_importance[0]]
        n_folds = len(fold_var_importance)
        n_vars = len(var_names)

        importance_matrix = np.zeros((n_folds, n_vars))
        for i, var_imp in enumerate(fold_var_importance):
            for j, item in enumerate(var_imp):
                importance_matrix[i, j] = item['total_importance']

        importance_matrix_norm = importance_matrix / importance_matrix.max(axis=1, keepdims=True)

        im = ax.imshow(importance_matrix_norm, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(n_vars))
        ax.set_yticks(np.arange(n_folds))
        ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=11, fontfamily='Times New Roman')
        ax.set_yticklabels([f'Fold {i + 1}' for i in range(n_folds)], fontsize=11, fontfamily='Times New Roman')
        ax.set_xlabel('Biomechanical Feature', fontsize=14, fontfamily='Times New Roman')
        ax.set_ylabel('Fold', fontsize=14, fontfamily='Times New Roman')
        ax.set_title(f'Variable Importance Across Folds (Normalized)',
                     fontsize=16, fontweight='bold', fontfamily='Times New Roman')

        cbar = plt.colorbar(im, ax=ax, label='Normalized Importance')
        cbar.ax.tick_params(labelsize=11)

        plt.tight_layout()

        base_path = os.path.join(output_dir, "fold_comparison")
        save_figure(fig, base_path)
        plt.close()

    def plot_combined_crossfold_importance(self):
        print("\n" + "=" * 80)
        print("Plotting combined cross-fold importance")
        print("=" * 80)

        if not self.all_crossfold_data:
            print("No cross-fold data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes_flat = axes.flatten()

        subplot_labels = ['(a)', '(b)', '(c)', '(d)']

        title_mapping = {
            '40_vs_80': 'Task 1 vs Task 2',
            'F40_vs_F80': 'Task 3 vs Task 4',
            '40_vs_F40': 'Task 1 vs Task 3',
            '80_vs_F80': 'Task 2 vs Task 4'
        }

        for idx, (ax, pair_name, label) in enumerate(
                zip(axes_flat, list(self.all_crossfold_data.keys()), subplot_labels)):
            data = self.all_crossfold_data[pair_name]
            var_importance_stats = data['var_importance_stats']

            var_names = var_importance_stats['var_names']
            mean_imp = var_importance_stats['mean']
            ci_lower = var_importance_stats['ci_lower']
            ci_upper = var_importance_stats['ci_upper']

            y_pos = np.arange(len(var_names))

            bars = ax.barh(y_pos, mean_imp, color=BAR_COLORS[:len(var_names)],
                           alpha=0.85, edgecolor='#2C3E50', linewidth=1.2)

            for i, (bar, lower, upper) in enumerate(zip(bars, ci_lower, ci_upper)):
                ax.plot([lower, upper], [bar.get_y() + bar.get_height() / 2,
                                         bar.get_y() + bar.get_height() / 2],
                        color='#2C3E50', linewidth=2, marker='|', markersize=8)

                label_y_top = bar.get_y() + bar.get_height() * 1.15

                ax.text(mean_imp[i] + max(mean_imp) * 0.02,
                        label_y_top,
                        f'{mean_imp[i]:.2f}', va='center', ha='left',
                        fontsize=12, fontfamily='Times New Roman', color='#2C3E50')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(var_names, fontsize=13, fontfamily='Times New Roman')
            ax.set_xlabel('Total SHAP Importance', fontsize=15, fontfamily='Times New Roman')
            ax.set_ylabel('Biomechanical Feature', fontsize=15, fontfamily='Times New Roman')

            display_title = title_mapping.get(pair_name, data['title'])

            ax.set_title(f'{display_title}',
                         fontsize=17, fontweight='bold', fontfamily='Times New Roman', pad=15)

            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')

            ax.annotate(label, xy=(-0.05, 1.02), xycoords='axes fraction',
                        fontsize=19, fontweight='bold', va='center', ha='left',
                        fontfamily='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        output_path = os.path.join(output_root, 'combined_crossfold_importance')
        save_figure(fig, output_path)
        plt.show()

        print(f"\nCombined cross-fold importance plot saved: {output_path}.png/.svg")

    def _save_crossfold_results(self, accuracy_stats, var_importance_stats,
                                shap_stability, pair_title, output_dir):
        report_path = os.path.join(output_dir, "crossfold_analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Cross-Fold SHAP Analysis Report - {pair_title}\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. Cross-Validation Settings\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of cross-validation folds: {self.n_folds}\n")
            f.write(f"Train:Test ratio = {self.n_folds - 1}:1\n\n")

            f.write("2. Model Accuracy Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean: {accuracy_stats['mean']:.4f}\n")
            f.write(f"Std: {accuracy_stats['std']:.4f}\n")
            f.write(f"95% CI: [{accuracy_stats['ci_lower']:.4f}, {accuracy_stats['ci_upper']:.4f}]\n")
            f.write(f"Min: {accuracy_stats['min']:.4f}\n")
            f.write(f"Max: {accuracy_stats['max']:.4f}\n")
            f.write(f"Per-fold accuracy: {', '.join([f'{acc:.4f}' for acc in accuracy_stats['all']])}\n\n")

            f.write("3. Variable Importance Statistics\n")
            f.write("-" * 30 + "\n")
            var_names = var_importance_stats['var_names']
            mean_imp = var_importance_stats['mean']
            std_imp = var_importance_stats['std']
            ci_lower = var_importance_stats['ci_lower']
            ci_upper = var_importance_stats['ci_upper']

            f.write(f"{'Variable':<20} {'Mean':<12} {'Std':<12} {'95% CI'}\n")
            f.write("-" * 70 + "\n")
            for i, name in enumerate(var_names):
                f.write(f"{name:<20} {mean_imp[i]:<12.4f} {std_imp[i]:<12.4f} "
                        f"[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]\n")

            f.write("\n4. SHAP Stability\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall stability metric: {shap_stability['overall_stability']:.4f}\n")
            f.write(f"Mean std: {shap_stability['mean_std']:.4f}\n")

            f.write("\n5. File Information\n")
            f.write("-" * 30 + "\n")
            f.write(f"Cross-validation results plot: crossfold_results.png/svg\n")
            f.write(f"Variable importance with CI plot: variable_importance_with_ci.png/svg\n")
            f.write(f"Fold comparison heatmap: fold_comparison.png/svg\n")
            f.write(f"Detailed report: crossfold_analysis_report.txt\n")

        print(f"Cross-validation analysis report saved: {report_path}")


def main():
    print("=" * 80)
    print("SHAP Cross-Validation Analysis - 5-Fold Cross-Validation")
    print("Train:Test ratio = 4:1")
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

    print("\n3. Plotting combined cross-fold importance...")
    crossfold_analyzer.plot_combined_crossfold_importance()

    print("\n" + "=" * 80)
    print("Cross-Validation SHAP Analysis Complete!")
    print("=" * 80)

    print(f"\nAll results saved to: {output_root}/")
    print("   - Each comparison has its own CrossFold folder")
    print("   - All figures saved as PNG, SVG, and PDF formats")
    print("   - Includes per-fold accuracy, variable importance with CI, and stability analysis")
    print("   - Combined cross-fold importance plot: combined_crossfold_importance.png/svg")

    return all_results


if __name__ == "__main__":
    results = main()