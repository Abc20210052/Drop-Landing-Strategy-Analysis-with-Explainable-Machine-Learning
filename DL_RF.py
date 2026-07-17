import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import os

warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

output_root = 'RandomForest_Analysis_Results_4to1'
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"Created main output directory: {output_root}")

def save_figure(fig, filename, dpi=300):
    os.makedirs(output_root, exist_ok=True)
    base_path = os.path.join(output_root, filename)

    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved PNG: {png_path}")

    svg_path = f"{base_path}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  Saved SVG: {svg_path}")

    pdf_path = f"{base_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved PDF: {pdf_path}")

file_paths = [
    r'F:\DL_40.csv',
    r'F:\DL_80.csv',
    r'F:\DL_P40.csv',
    r'F:\DL_P80.csv'
]

class_labels = ['40', '80', 'F40', 'F80']

def load_and_prepare_data():
    all_data = []
    all_labels = []
    all_subject_ids = []
    all_repetition_ids = []

    for i, file_path in enumerate(file_paths):
        if os.path.exists(file_path):
            print("Loading: {}".format(file_path))
            df = pd.read_csv(file_path)
            print("  Data shape: {}".format(df.shape))

            for j in range(len(df)):
                subject_num = j // 3
                repetition_num = j % 3 + 1
                subject_id = "Class{}_Subject{}".format(i, subject_num)
                repetition_id = "Rep{}".format(repetition_num)
                all_data.append(df.iloc[j].values)
                all_labels.append(i)
                all_subject_ids.append(subject_id)
                all_repetition_ids.append(repetition_id)
        else:
            print("File not found: {}".format(file_path))

    X = np.array(all_data)
    y = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)
    repetition_ids = np.array(all_repetition_ids)

    print("\nTotal samples: {}".format(len(X)))
    print("Feature dimension: {}".format(X.shape[1]))
    print("Class distribution: {}".format(np.bincount(y)))
    print("Number of subjects: {}".format(len(np.unique(subject_ids))))
    print("Number of repetitions per subject: 3")

    print("\n" + "=" * 60)
    print("Performing Z-score Standardization (Global)")
    print("=" * 60)

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"   Found {nan_count} NaN values, filling with column mean")
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        for i, j in zip(nan_indices[0], nan_indices[1]):
            X[i, j] = col_means[j]

    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        print(f"   Found {inf_count} Inf values, replacing with finite values")
        X[np.isinf(X)] = np.finfo(np.float64).max

    print("\nApplying Z-score standardization to all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nZ-score Standardization Statistics:")
    print(f"   Overall mean: {np.mean(X_scaled):.6f} (should be close to 0)")
    print(f"   Overall std: {np.std(X_scaled):.6f} (should be close to 1)")

    print("\nStandardization check by group:")
    unique_groups = np.unique(y)
    for group_idx in unique_groups:
        mask = (y == group_idx)
        group_data = X_scaled[mask]
        group_name = class_labels[group_idx]
        print(f"   {group_name}: mean = {np.mean(group_data):.4f}, std = {np.std(group_data):.4f}")

    print("\nZ-score standardization completed successfully!")
    print("All features scaled to mean=0, std=1")

    return X_scaled, y, subject_ids, repetition_ids, class_labels

def create_cross_subject_split_4_to_1(X, y, subject_ids, test_size=0.2):
    unique_subjects = np.unique(subject_ids)

    subject_info = {}
    for subj in unique_subjects:
        subj_indices = np.where(subject_ids == subj)[0]
        subj_class = y[subj_indices[0]]
        subject_info[subj] = {
            'indices': subj_indices,
            'class': subj_class
        }

    class_subjects = {}
    for subj, info in subject_info.items():
        class_id = info['class']
        if class_id not in class_subjects:
            class_subjects[class_id] = []
        class_subjects[class_id].append(subj)

    train_subjects = []
    test_subjects = []

    for class_id, subjects in class_subjects.items():
        n_subjects = len(subjects)
        n_test = int(np.ceil(n_subjects * test_size))
        np.random.shuffle(subjects)
        test_subjects.extend(subjects[:n_test])
        train_subjects.extend(subjects[n_test:])

    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

    print("\nSubject-based 4:1 Split Details:")
    print("-" * 40)
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Training subjects: {len(train_subjects)} (80%)")
    print(f"Test subjects: {len(test_subjects)} (20%)")

    for class_id, subjects in class_subjects.items():
        class_name = class_labels[class_id] if class_id < len(class_labels) else f"Class_{class_id}"
        train_count = len([s for s in train_subjects if subject_info[s]['class'] == class_id])
        test_count = len([s for s in test_subjects if subject_info[s]['class'] == class_id])
        print(f"\n{class_name}:")
        print(f"  Total subjects: {len(subjects)}")
        print(f"  Training subjects: {train_count}")
        print(f"  Test subjects: {test_count}")

    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Train/Test ratio: {len(train_indices)}:{len(test_indices)} ≈ {len(train_indices) / len(test_indices):.2f}:1")

    return train_indices, test_indices

def plot_confusion_matrix_percent_only(cm, class_names, title, base_filename):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    annot_matrix = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            annotation = "{:.1f}%".format(cm_percent[i, j])
            row.append(annotation)
        annot_matrix.append(row)

    sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Blues',
                cbar=True, square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 12, 'va': 'center', 'fontweight': 'bold'},
                cbar_kws={'label': 'Accuracy (%)'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, base_filename)
    plt.show()

def plot_cross_validation_confusion_matrices(results, class_labels_pair, base_filename):
    cv_folds = results.get('cv_folds_details', [])
    if not cv_folds:
        print("Warning: Cross-validation details not found")
        return

    n_folds = len(cv_folds)
    n_cols = min(3, n_folds)
    n_rows = (n_folds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.cm.Blues

    for fold_idx, fold_data in enumerate(cv_folds):
        row = fold_idx // n_cols
        col = fold_idx % n_cols
        ax = axes[row, col] if n_rows > 1 or n_cols > 1 else axes[col]

        cm = fold_data['confusion_matrix']
        fold_accuracy = fold_data['accuracy'] * 100

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annot_matrix = []
        for i in range(cm.shape[0]):
            row_annot = []
            for j in range(cm.shape[1]):
                annotation = "{:.1f}%".format(cm_percent[i, j])
                row_annot.append(annotation)
            annot_matrix.append(row_annot)

        sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap=cmap,
                    cbar=False, square=True,
                    xticklabels=class_labels_pair,
                    yticklabels=class_labels_pair,
                    annot_kws={'size': 10, 'va': 'center', 'fontweight': 'bold'},
                    ax=ax)

        ax.set_title(f'Fold {fold_idx + 1}\nAccuracy: {fold_accuracy:.1f}%',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    for fold_idx in range(len(cv_folds), n_rows * n_cols):
        row = fold_idx // n_cols
        col = fold_idx % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')

    class_pair_name = results['class_pair']
    fig.suptitle(
        f'Cross-Validation Confusion Matrices (Accuracy %)\n{class_pair_name}\n(Z-score Standardized, 5-Fold CV, 4:1 Split)',
        fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, base_filename)
    plt.show()

def plot_average_cv_confusion_matrix(results, class_labels_pair, base_filename):
    cv_folds = results.get('cv_folds_details', [])
    if not cv_folds:
        print("Warning: Cross-validation details not found")
        return

    all_cm_percent = []
    for fold in cv_folds:
        cm = fold['confusion_matrix']
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        all_cm_percent.append(cm_percent)

    avg_cm_percent = np.mean(all_cm_percent, axis=0)
    std_cm_percent = np.std(all_cm_percent, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    annot_matrix = []
    for i in range(avg_cm_percent.shape[0]):
        row_annot = []
        for j in range(avg_cm_percent.shape[1]):
            annotation = "{:.1f}%".format(avg_cm_percent[i, j])
            row_annot.append(annotation)
        annot_matrix.append(row_annot)

    sns.heatmap(avg_cm_percent, annot=annot_matrix, fmt='', cmap='Blues',
                cbar=True, square=True,
                xticklabels=class_labels_pair,
                yticklabels=class_labels_pair,
                annot_kws={'size': 11, 'va': 'center', 'fontweight': 'bold'},
                cbar_kws={'label': 'Accuracy (%)'},
                ax=ax1)

    ax1.set_title(f'Cross-Validation Confusion Matrix\n{results["class_pair"]}\n(Accuracy %, 4:1 Split)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    std_annot_matrix = []
    for i in range(std_cm_percent.shape[0]):
        row_annot = []
        for j in range(std_cm_percent.shape[1]):
            annotation = "{:.1f}%".format(std_cm_percent[i, j])
            row_annot.append(annotation)
        std_annot_matrix.append(row_annot)

    sns.heatmap(std_cm_percent, annot=std_annot_matrix, fmt='', cmap='Oranges',
                cbar=True, square=True,
                xticklabels=class_labels_pair,
                yticklabels=class_labels_pair,
                annot_kws={'size': 11, 'va': 'center', 'fontweight': 'bold'},
                cbar_kws={'label': 'Std Deviation (%)'},
                ax=ax2)

    ax2.set_title('Accuracy Standard Deviation\nAcross 5 Folds (%, 4:1 Split)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    fold_accuracies = [fold['accuracy'] * 100 for fold in cv_folds]
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    plt.tight_layout()
    save_figure(fig, base_filename)
    plt.show()

def save_fold_results_to_csv_rf(all_results, n_estimators):
    print("\n" + "=" * 60)
    print("Saving Cross-Validation Fold Results to CSV (Random Forest)")
    print("=" * 60)

    fold_summary_rows = []

    for result in all_results:
        class_pair = result['class_pair']
        class1_name = result['class1_name']
        class2_name = result['class2_name']

        for fold_info in result['cv_folds_details']:
            row = {
                'Classification_Task': class_pair,
                'Class1': class1_name,
                'Class2': class2_name,
                'Fold': fold_info['fold'],
                'Train_Subjects': fold_info['train_subjects'],
                'Val_Subjects': fold_info['val_subjects'],
                'Train_Samples': fold_info['train_samples'],
                'Val_Samples': fold_info['val_samples'],
                'Accuracy': fold_info['accuracy'],
                'F1_Score': fold_info['f1_score'],
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_summary_rows.append(row)

    all_folds_df = pd.DataFrame(fold_summary_rows)
    all_folds_csv_path = os.path.join(output_root, f'RF_all_folds_detailed_4to1_trees{n_estimators}.csv')
    all_folds_df.to_csv(all_folds_csv_path, index=False)
    print(f"  Saved all folds detailed results to: {all_folds_csv_path}")

    summary_rows = []
    for result in all_results:
        row = {
            'Classification_Task': result['class_pair'],
            'Class1': result['class1_name'],
            'Class2': result['class2_name'],
            'N_Estimators': n_estimators,
            'Train_Accuracy': result['train_accuracy'],
            'Train_F1': result['train_f1'],
            'Test_Accuracy': result['test_accuracy'],
            'Test_F1': result['test_f1'],
            'ROC_AUC': result['roc_auc'],
            'CV_Mean_Accuracy': result['cv_mean'],
            'CV_Std_Accuracy': result['cv_std'],
            'CV_Mean_F1': result['cv_f1_mean'],
            'CV_Std_F1': result['cv_f1_std'],
            'Fold1_Accuracy': result['cv_scores'][0] if len(result['cv_scores']) > 0 else None,
            'Fold2_Accuracy': result['cv_scores'][1] if len(result['cv_scores']) > 1 else None,
            'Fold3_Accuracy': result['cv_scores'][2] if len(result['cv_scores']) > 2 else None,
            'Fold4_Accuracy': result['cv_scores'][3] if len(result['cv_scores']) > 3 else None,
            'Fold5_Accuracy': result['cv_scores'][4] if len(result['cv_scores']) > 4 else None,
            'Fold1_F1': result['cv_f1_scores'][0] if len(result['cv_f1_scores']) > 0 else None,
            'Fold2_F1': result['cv_f1_scores'][1] if len(result['cv_f1_scores']) > 1 else None,
            'Fold3_F1': result['cv_f1_scores'][2] if len(result['cv_f1_scores']) > 2 else None,
            'Fold4_F1': result['cv_f1_scores'][3] if len(result['cv_f1_scores']) > 3 else None,
            'Fold5_F1': result['cv_f1_scores'][4] if len(result['cv_f1_scores']) > 4 else None,
            'Avg_Tree_Depth': result['avg_tree_depth'],
            'Max_Tree_Depth': result['max_tree_depth'],
            'Top_Feature_Importance': result['top_feature_importance'],
            'OOB_Score': result.get('oob_score', None),
            'Standardization': result['standardization_method'],
            'Data_Split': result['data_split']
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_root, f'RF_overall_summary_4to1_trees{n_estimators}.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved overall summary to: {summary_csv_path}")

    for result in all_results:
        class_pair = result['class_pair']
        fold_rows = []
        for fold_info in result['cv_folds_details']:
            row = {
                'Fold': fold_info['fold'],
                'Train_Subjects': fold_info['train_subjects'],
                'Val_Subjects': fold_info['val_subjects'],
                'Train_Samples': fold_info['train_samples'],
                'Val_Samples': fold_info['val_samples'],
                'Accuracy': fold_info['accuracy'],
                'F1_Score': fold_info['f1_score'],
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        filename = f'RF_fold_results_{class_pair}_4to1_trees{n_estimators}.csv'
        fold_csv_path = os.path.join(output_root, filename)
        fold_df.to_csv(fold_csv_path, index=False)
        print(f"  Saved fold results for {class_pair} to: {fold_csv_path}")

    print("\nAll cross-validation fold results saved to CSV files")

def evaluate_two_class_classification_rf(class1, class2, X, y, subject_ids, class_labels, n_estimators=150, max_depth=None):
    print("\n" + "=" * 60)
    print("Random Forest Classification: {} vs {}".format(class_labels[class1], class_labels[class2]))
    print("Random Forest with {} trees, max_depth={}".format(n_estimators, max_depth))
    print("4:1 Subject-based Split")
    print("=" * 60)

    mask = (y == class1) | (y == class2)
    X_binary = X[mask]
    y_binary = y[mask]
    subject_ids_binary = subject_ids[mask]

    y_binary = np.where(y_binary == class1, 0, 1)

    print("Total samples: {}".format(len(X_binary)))
    print("Class distribution: {}: {}, {}: {}".format(
        class_labels[class1], np.sum(y_binary == 0),
        class_labels[class2], np.sum(y_binary == 1)))

    unique_subjects = np.unique(subject_ids_binary)
    class0_subjects = []
    class1_subjects = []

    for subj in unique_subjects:
        subj_indices = np.where(subject_ids_binary == subj)[0]
        subj_class = y_binary[subj_indices[0]]
        if subj_class == 0:
            class0_subjects.append(subj)
        else:
            class1_subjects.append(subj)

    print("Unique subjects - {}: {}, {}: {}".format(
        class_labels[class1], len(class0_subjects),
        class_labels[class2], len(class1_subjects)))

    train_indices, test_indices = create_cross_subject_split_4_to_1(
        X_binary, y_binary, subject_ids_binary, test_size=0.2
    )

    X_train, X_test = X_binary[train_indices], X_binary[test_indices]
    y_train, y_test = y_binary[train_indices], y_binary[test_indices]

    print("\nData Standardization Status:")
    print("Data was already standardized during loading (Z-score)")
    print("Features have mean ≈ 0 and std ≈ 1")
    print("Random Forest can handle features with different scales, but standardization helps")

    X_train_scaled = X_train
    X_test_scaled = X_test

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    y_train_pred = rf_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='binary')

    y_test_pred = rf_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='binary')

    y_test_proba = rf_model.predict_proba(X_test_scaled)

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    else:
        roc_auc = 0.0

    unique_subjects = np.unique(subject_ids_binary)
    subject_labels_list = []

    for subj in unique_subjects:
        subj_indices = np.where(subject_ids_binary == subj)[0]
        subj_label = y_binary[subj_indices[0]]
        subject_labels_list.append(subj_label)

    cv_scores = []
    cv_f1_scores = []
    cv_folds_details = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_subjects = []
    for subj in unique_subjects:
        subj_indices = np.where(subject_ids_binary == subj)[0]
        subj_features = X_binary[subj_indices]
        subject_mean_features = np.mean(subj_features, axis=0)
        X_subjects.append(subject_mean_features)

    X_subjects = np.array(X_subjects)
    y_subjects = np.array(subject_labels_list)

    print("\n" + "-" * 60)
    print("5-Fold Cross-Validation Details (Subject-Level)")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_subjects, y_subjects), 1):
        train_subjects = unique_subjects[train_idx]
        val_subjects = unique_subjects[val_idx]

        train_mask = np.isin(subject_ids_binary, train_subjects)
        val_mask = np.isin(subject_ids_binary, val_subjects)

        X_cv_train, X_cv_val = X_binary[train_mask], X_binary[val_mask]
        y_cv_train, y_cv_val = y_binary[train_mask], y_binary[val_mask]

        rf_cv = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
        rf_cv.fit(X_cv_train, y_cv_train)

        y_cv_pred = rf_cv.predict(X_cv_val)
        cv_score = accuracy_score(y_cv_val, y_cv_pred)
        cv_f1 = f1_score(y_cv_val, y_cv_pred, average='binary')

        cv_scores.append(cv_score)
        cv_f1_scores.append(cv_f1)

        cv_cm = confusion_matrix(y_cv_val, y_cv_pred)

        cv_folds_details.append({
            'fold': fold,
            'accuracy': cv_score,
            'f1_score': cv_f1,
            'confusion_matrix': cv_cm,
            'train_subjects': len(train_subjects),
            'val_subjects': len(val_subjects),
            'train_samples': len(X_cv_train),
            'val_samples': len(X_cv_val),
            'cm_00': cv_cm[0, 0] if cv_cm.shape[0] > 0 else 0,
            'cm_01': cv_cm[0, 1] if cv_cm.shape[1] > 1 else 0,
            'cm_10': cv_cm[1, 0] if cv_cm.shape[0] > 1 else 0,
            'cm_11': cv_cm[1, 1] if cv_cm.shape[0] > 1 and cv_cm.shape[1] > 1 else 0
        })

        print("  Fold {}: Train subjects={}, Val subjects={}, Train samples={}, Val samples={}".format(
            fold, len(train_subjects), len(val_subjects),
            len(X_cv_train), len(X_cv_val)))
        print("    Accuracy = {:.4f}, F1 = {:.4f}".format(cv_score, cv_f1))
        print("    Confusion Matrix:")
        print("      [[{}  {}]".format(cv_cm[0, 0], cv_cm[0, 1] if cv_cm.shape[1] > 1 else 0))
        print("       [{}  {}]]".format(cv_cm[1, 0] if cv_cm.shape[0] > 1 else 0,
                                        cv_cm[1, 1] if cv_cm.shape[0] > 1 and cv_cm.shape[1] > 1 else 0))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_f1_mean = np.mean(cv_f1_scores)
    cv_f1_std = np.std(cv_f1_scores)

    print("-" * 60)
    print("Cross-Validation Summary:")
    print("  Mean Accuracy = {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("  Mean F1 Score = {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))
    print("  Individual Accuracy scores: {}".format([round(s, 4) for s in cv_scores]))
    print("  Individual F1 scores: {}".format([round(s, 4) for s in cv_f1_scores]))
    print("-" * 60)

    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("Results Summary ({} vs {})".format(class_labels[class1], class_labels[class2]))
    print("=" * 60)
    print("Training Accuracy: {:.4f} ({:.1f}%)".format(train_accuracy, train_accuracy * 100))
    print("Training F1 Score: {:.4f}".format(train_f1))
    print("Test Accuracy: {:.4f} ({:.1f}%)".format(test_accuracy, test_accuracy * 100))
    print("Test F1 Score: {:.4f}".format(test_f1))
    print("ROC AUC Score: {:.4f}".format(roc_auc))
    print("Cross-validation Mean Accuracy: {:.4f} (±{:.4f}) ({:.1f}% ±{:.1f}%)".format(
        cv_mean, cv_std, cv_mean * 100, cv_std * 100))
    print("Cross-validation Mean F1 Score: {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))

    print("\nRandom Forest Model Parameters:")
    print("  Number of trees: {}".format(rf_model.n_estimators))
    print("  Max depth: {}".format(rf_model.max_depth))
    print("  Min samples split: {}".format(rf_model.min_samples_split))
    print("  Min samples leaf: {}".format(rf_model.min_samples_leaf))
    print("  Max features: {}".format(rf_model.max_features))
    print("  Bootstrap: {}".format(rf_model.bootstrap))
    print("  Number of features: {}".format(rf_model.n_features_in_))

    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[::-1][:10]
    print("\nTop 10 Feature Importances:")
    for i, idx in enumerate(top_features_idx):
        print(f"  {i + 1:2d}. Feature {idx:3d}: {feature_importance[idx]:.6f}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=[class_labels[class1], class_labels[class2]]))

    print("Confusion Matrix (Counts):")
    print(cm)

    plot_title = "Random Forest: {} vs {}\n(4:1 Split)".format(
        class_labels[class1], class_labels[class2])
    cm_filename = f"rf_confusion_matrix_{class_labels[class1]}_vs_{class_labels[class2]}_4to1_trees{n_estimators}"
    plot_confusion_matrix_percent_only(cm,
                                       [class_labels[class1], class_labels[class2]],
                                       plot_title,
                                       cm_filename)

    print("\n" + "=" * 60)
    print("Generating Cross-Validation Confusion Matrices...")
    print("=" * 60)

    results_dict = {
        'class_pair': "{}_vs_{}".format(class_labels[class1], class_labels[class2]),
        'cv_folds_details': cv_folds_details
    }

    cv_cm_filename = f"rf_cv_confusion_matrices_{class_labels[class1]}_vs_{class_labels[class2]}_4to1_trees{n_estimators}"
    plot_cross_validation_confusion_matrices(results_dict,
                                             [class_labels[class1], class_labels[class2]],
                                             cv_cm_filename)

    avg_cv_cm_filename = f"rf_avg_cv_confusion_matrix_{class_labels[class1]}_vs_{class_labels[class2]}_4to1_trees{n_estimators}"
    plot_average_cv_confusion_matrix(results_dict,
                                     [class_labels[class1], class_labels[class2]],
                                     avg_cv_cm_filename)

    if hasattr(rf_model, 'oob_score_'):
        print("\nOut-of-Bag (OOB) Score: {:.4f} ({:.1f}%)".format(rf_model.oob_score_, rf_model.oob_score_ * 100))

    tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    print("\nTree Depth Analysis:")
    print("  Average tree depth: {:.2f}".format(np.mean(tree_depths)))
    print("  Max tree depth: {}".format(np.max(tree_depths)))
    print("  Min tree depth: {}".format(np.min(tree_depths)))

    results = {
        'class_pair': "{}_vs_{}".format(class_labels[class1], class_labels[class2]),
        'class1_name': class_labels[class1],
        'class2_name': class_labels[class2],
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_f1_mean': cv_f1_mean,
        'cv_f1_std': cv_f1_std,
        'cv_scores': cv_scores,
        'cv_f1_scores': cv_f1_scores,
        'cv_folds_details': cv_folds_details,
        'confusion_matrix': cm,
        'class_report': classification_report(y_test, y_test_pred,
                                              target_names=[class_labels[class1], class_labels[class2]],
                                              output_dict=True),
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'model': rf_model,
        'feature_importances': feature_importance,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'tree_depths': tree_depths,
        'avg_tree_depth': np.mean(tree_depths),
        'max_tree_depth': np.max(tree_depths),
        'top_feature_importance': np.max(feature_importance),
        'standardization_method': 'Z-score (StandardScaler)',
        'oob_score': rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') else None,
        'data_split': '4:1 Subject-based',
        'split_ratio': '80% Training, 20% Testing'
    }

    return results

def plot_rf_feature_importance(results, n_features_to_plot=20, base_filename=None):
    feature_importance = results['feature_importances']
    class_pair = results['class_pair']

    sorted_idx = np.argsort(feature_importance)[::-1]
    top_n = min(n_features_to_plot, len(feature_importance))

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(top_n)
    ax.barh(y_pos, feature_importance[sorted_idx[:top_n]], align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Feature {i}' for i in sorted_idx[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Random Forest Feature Importance\n{class_pair}\n(Top {top_n} Features, 4:1 Split)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(feature_importance[sorted_idx[:top_n]]):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    cumulative_importance = np.cumsum(feature_importance[sorted_idx])
    ax2 = ax.twiny()
    ax2.plot(cumulative_importance[:top_n], y_pos, 'darkblue', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Cumulative Importance', color='darkblue', fontsize=12)
    ax2.tick_params(axis='x', labelcolor='darkblue')

    threshold_80 = np.where(cumulative_importance >= 0.8)[0]
    threshold_90 = np.where(cumulative_importance >= 0.9)[0]

    if len(threshold_80) > 0:
        n_features_80 = threshold_80[0] + 1
        ax2.axvline(x=0.8, color='blue', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.95, f'80% importance with {n_features_80} features',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    if len(threshold_90) > 0:
        n_features_90 = threshold_90[0] + 1
        ax2.axvline(x=0.9, color='navy', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.90, f'90% importance with {n_features_90} features',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.5))

    plt.tight_layout()

    if base_filename:
        save_figure(fig, base_filename)

    plt.show()

def plot_rf_probability_distribution(results, class_labels_pair, base_filename):
    y_true = results['y_true']
    y_proba = results['y_proba']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['blue', 'steelblue']
    for class_idx, class_name in enumerate(class_labels_pair):
        class_mask = (y_true == class_idx)
        class_proba = y_proba[class_mask, class_idx]

        axes[0].hist(class_proba, bins=20, alpha=0.6, label='True {}'.format(class_name),
                     density=True, edgecolor='black', color=colors[class_idx])

    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Random Forest Probability Distribution\nby True Class (4:1 Split)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    max_proba = np.max(y_proba, axis=1)
    prediction_certainty = max_proba

    axes[1].hist(prediction_certainty, bins=20, alpha=0.7, color='steelblue',
                 density=True, edgecolor='black')

    avg_certainty = np.mean(prediction_certainty)
    axes[1].axvline(x=avg_certainty, color='darkblue', linestyle='--', linewidth=2,
                    label='Avg Certainty: {:.3f}'.format(avg_certainty))

    correct_mask = (y_true == results['y_pred'])
    correct_certainty = prediction_certainty[correct_mask]
    incorrect_certainty = prediction_certainty[~correct_mask]

    if len(correct_certainty) > 0:
        axes[1].axvline(x=np.mean(correct_certainty), color='blue', linestyle='--', linewidth=1.5,
                        label='Avg Correct: {:.3f}'.format(np.mean(correct_certainty)))

    if len(incorrect_certainty) > 0:
        axes[1].axvline(x=np.mean(incorrect_certainty), color='lightblue', linestyle='--', linewidth=1.5,
                        label='Avg Incorrect: {:.3f}'.format(np.mean(incorrect_certainty)))

    axes[1].set_xlabel('Prediction Certainty (Max Probability)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1].set_title('Random Forest Prediction Certainty (4:1 Split)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if base_filename:
        save_figure(fig, base_filename)

    plt.show()

def plot_rf_trees_analysis(results_list, base_filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    class_pairs = [r['class_pair'] for r in results_list]
    tree_depths_list = [r['tree_depths'] for r in results_list]
    test_accuracies = [r['test_accuracy'] for r in results_list]
    test_f1_scores = [r['test_f1'] for r in results_list]
    roc_auc_scores = [r.get('roc_auc', 0) for r in results_list]

    colors = ['blue', 'steelblue', 'lightblue', 'darkblue']
    for i, (depths, class_pair) in enumerate(zip(tree_depths_list, class_pairs)):
        if i < len(colors):
            axes[0].hist(depths, bins=20, alpha=0.5, label=class_pair, density=True, color=colors[i])

    axes[0].set_xlabel('Tree Depth', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Random Forest Tree Depth Distribution\nAcross Classification Tasks (4:1 Split)',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    avg_tree_depths = [np.mean(depths) for depths in tree_depths_list]
    axes[1].scatter(avg_tree_depths, test_accuracies, s=100, alpha=0.7, color='blue')

    for i, (depth, acc, class_pair) in enumerate(zip(avg_tree_depths, test_accuracies, class_pairs)):
        axes[1].text(depth, acc, class_pair, fontsize=9, ha='center', va='bottom')

    axes[1].set_xlabel('Average Tree Depth', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy vs. Average Tree Depth (4:1 Split)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if len(avg_tree_depths) > 1:
        z = np.polyfit(avg_tree_depths, test_accuracies, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(avg_tree_depths), max(avg_tree_depths), 100)
        axes[1].plot(x_range, p(x_range), 'darkblue', linestyle='--', alpha=0.7,
                     label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        axes[1].legend()

    axes[2].scatter(test_f1_scores, roc_auc_scores, s=100, alpha=0.7, color='steelblue', edgecolor='black')

    for i, (f1, auc, class_pair) in enumerate(zip(test_f1_scores, roc_auc_scores, class_pairs)):
        axes[2].text(f1, auc, class_pair, fontsize=9, ha='center', va='bottom')

    axes[2].set_xlabel('Test F1 Score', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    axes[2].set_title('ROC AUC vs. Test F1 (4:1 Split)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance Level')
    axes[2].legend()

    n_tasks = len(results_list)
    if n_tasks > 1:
        all_importances = np.array([r['feature_importances'] for r in results_list])
        correlation_matrix = np.corrcoef(all_importances)

        im = axes[3].imshow(correlation_matrix, cmap='Blues', vmin=0, vmax=1)

        cbar = fig.colorbar(im, ax=axes[3])
        cbar.set_label('Correlation', fontsize=12)

        axes[3].set_xticks(np.arange(n_tasks))
        axes[3].set_yticks(np.arange(n_tasks))
        axes[3].set_xticklabels(class_pairs, rotation=45, ha='right')
        axes[3].set_yticklabels(class_pairs)

        for i in range(n_tasks):
            for j in range(n_tasks):
                text = axes[3].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                    ha="center", va="center", color="white", fontsize=9)

        axes[3].set_title('Feature Importance Correlation\nBetween Classification Tasks (4:1 Split)',
                          fontsize=14, fontweight='bold')

    plt.tight_layout()

    if base_filename:
        save_figure(fig, base_filename)

    plt.show()

def plot_comprehensive_results_rf(all_results, n_estimators, base_filename):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    class_pairs = [result['class_pair'] for result in all_results]
    test_accuracies = [result['test_accuracy'] for result in all_results]
    test_f1_scores = [result['test_f1'] for result in all_results]
    train_accuracies = [result['train_accuracy'] for result in all_results]
    train_f1_scores = [result['train_f1'] for result in all_results]
    roc_auc_scores = [result['roc_auc'] for result in all_results]
    cv_means = [result['cv_mean'] for result in all_results]
    cv_stds = [result['cv_std'] for result in all_results]
    cv_f1_means = [result['cv_f1_mean'] for result in all_results]
    cv_f1_stds = [result['cv_f1_std'] for result in all_results]

    avg_tree_depths = [np.mean(result['tree_depths']) for result in all_results]
    top_feature_importances = [np.max(result['feature_importances']) for result in all_results]

    x = np.arange(len(class_pairs))
    width = 0.25

    colors = ['lightblue', 'steelblue', 'blue']

    axes[0].bar(x - width, train_accuracies, width, label='Training Accuracy', color=colors[0], alpha=0.8, edgecolor='black')
    axes[0].bar(x, test_accuracies, width, label='Test Accuracy', color=colors[1], alpha=0.8, edgecolor='black')
    axes[0].bar(x + width, cv_means, width, label='CV Accuracy', color=colors[2], alpha=0.8,
                yerr=cv_stds, capsize=5, edgecolor='black')

    axes[0].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Random Forest ({} trees) Accuracy Comparison\n(4:1 Split, Z-score Standardized)'.format(n_estimators),
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_pairs, rotation=45, ha='right', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])

    for i, (train_acc, test_acc, cv_acc) in enumerate(zip(train_accuracies, test_accuracies, cv_means)):
        axes[0].text(i - width, train_acc + 0.02, '{:.3f}'.format(train_acc),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[0].text(i, test_acc + 0.02, '{:.3f}'.format(test_acc),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[0].text(i + width, cv_acc + 0.02, '{:.3f}'.format(cv_acc),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    colors_f1 = ['#D8BFD8', '#9370DB', '#6A0DAD']

    axes[1].bar(x - width, train_f1_scores, width, label='Training F1', color=colors_f1[0], alpha=0.8, edgecolor='black')
    axes[1].bar(x, test_f1_scores, width, label='Test F1', color=colors_f1[1], alpha=0.8, edgecolor='black')
    axes[1].bar(x + width, cv_f1_means, width, label='CV F1', color=colors_f1[2], alpha=0.8,
                yerr=cv_f1_stds, capsize=5, edgecolor='black')

    axes[1].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Random Forest ({} trees) F1 Score Comparison\n(4:1 Split, Z-score Standardized)'.format(n_estimators),
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_pairs, rotation=45, ha='right', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    for i, (train_f1, test_f1, cv_f1) in enumerate(zip(train_f1_scores, test_f1_scores, cv_f1_means)):
        axes[1].text(i - width, train_f1 + 0.02, '{:.3f}'.format(train_f1),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1].text(i, test_f1 + 0.02, '{:.3f}'.format(test_f1),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1].text(i + width, cv_f1 + 0.02, '{:.3f}'.format(cv_f1),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    x_pos = np.arange(len(class_pairs))
    width = 0.35

    depth_bars = axes[2].bar(x_pos - width / 2, avg_tree_depths, width,
                             label='Average Tree Depth', color='steelblue', alpha=0.8, edgecolor='black')

    ax2 = axes[2].twinx()
    feature_bars = ax2.bar(x_pos + width / 2, top_feature_importances, width,
                           label='Top Feature Importance', color='lightblue', alpha=0.8, edgecolor='black')

    axes[2].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Tree Depth', fontsize=12, fontweight='bold', color='steelblue')
    ax2.set_ylabel('Top Feature Importance', fontsize=12, fontweight='bold', color='lightblue')
    axes[2].set_title('Tree Depth & Feature Importance Analysis (4:1 Split)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_pairs, rotation=45, ha='right', fontweight='bold')

    axes[2].tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='lightblue')

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    axes[2].grid(True, alpha=0.3)

    for i, (depth, importance) in enumerate(zip(avg_tree_depths, top_feature_importances)):
        axes[2].text(i - width / 2, depth + 0.5, '{:.1f}'.format(depth),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='steelblue')
        ax2.text(i + width / 2, importance + 0.005, '{:.4f}'.format(importance),
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='lightblue')

    summary_data = []
    for result in all_results:
        cm = result['confusion_matrix']
        if cm.shape[0] == 2:
            tp = cm[0, 0]
            fn = cm[0, 1]
            fp = cm[1, 0]
            tn = cm[1, 1]

            recall_class0 = tp / max(tp + fn, 1)
            precision_class0 = tp / max(tp + fp, 1)
            specificity_class0 = tn / max(tn + fp, 1)
            f1_class0 = 2 * (precision_class0 * recall_class0) / max(precision_class0 + recall_class0, 1e-10)

            summary_data.append([recall_class0, precision_class0, specificity_class0, f1_class0])

    if summary_data:
        summary_data = np.array(summary_data)
        x_pos = np.arange(len(class_pairs))
        width = 0.2

        colors_metrics = ['lightblue', 'steelblue', 'blue', 'darkblue']

        axes[3].bar(x_pos - 1.5 * width, summary_data[:, 0], width, label='Recall',
                    color=colors_metrics[0], alpha=0.8, edgecolor='black')
        axes[3].bar(x_pos - 0.5 * width, summary_data[:, 1], width, label='Precision',
                    color=colors_metrics[1], alpha=0.8, edgecolor='black')
        axes[3].bar(x_pos + 0.5 * width, summary_data[:, 2], width, label='Specificity',
                    color=colors_metrics[2], alpha=0.8, edgecolor='black')
        axes[3].bar(x_pos + 1.5 * width, summary_data[:, 3], width, label='F1-Score',
                    color=colors_metrics[3], alpha=0.8, edgecolor='black')

        axes[3].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[3].set_title('Class 0 Performance Metrics\n(Random Forest, {} trees, 4:1 Split)'.format(n_estimators),
                          fontsize=14, fontweight='bold')
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(class_pairs, rotation=45, ha='right', fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.1])

        for i, (recall, precision, specificity, f1) in enumerate(summary_data):
            axes[3].text(i - 1.5 * width, recall + 0.02, '{:.3f}'.format(recall),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axes[3].text(i - 0.5 * width, precision + 0.02, '{:.3f}'.format(precision),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axes[3].text(i + 0.5 * width, specificity + 0.02, '{:.3f}'.format(specificity),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axes[3].text(i + 1.5 * width, f1 + 0.02, '{:.3f}'.format(f1),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    if base_filename:
        save_figure(fig, base_filename)

    plt.show()

def analyze_rf_performance(all_results, n_estimators):
    print("\n" + "=" * 80)
    print("RANDOM FOREST CLASSIFIER SPECIFIC ANALYSIS ({} trees)".format(n_estimators))
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 80)

    for result in all_results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_proba = result['y_proba']
        feature_importance = result['feature_importances']
        tree_depths = result['tree_depths']

        correct_mask = (y_true == y_pred)

        max_proba = np.max(y_proba, axis=1)
        avg_certainty_correct = np.mean(max_proba[correct_mask]) if np.sum(correct_mask) > 0 else 0
        avg_certainty_incorrect = np.mean(max_proba[~correct_mask]) if np.sum(~correct_mask) > 0 else 0

        importance_threshold = 0.01
        n_important_features = np.sum(feature_importance > importance_threshold)
        top_5_importance = np.sum(np.sort(feature_importance)[::-1][:5])

        print("\n{}:".format(result['class_pair']))
        print("  F1 Scores:")
        print("    Training F1: {:.4f}".format(result['train_f1']))
        print("    Test F1: {:.4f}".format(result['test_f1']))
        print("    CV F1: {:.4f} (±{:.4f})".format(result['cv_f1_mean'], result['cv_f1_std']))

        print("\n  Prediction Certainty Analysis:")
        print("    Correct predictions - Avg certainty: {:.4f}".format(avg_certainty_correct))
        print("    Incorrect predictions - Avg certainty: {:.4f}".format(avg_certainty_incorrect))

        if avg_certainty_incorrect > 0 and avg_certainty_correct > 0:
            certainty_ratio = avg_certainty_incorrect / avg_certainty_correct
            if certainty_ratio > 1.0:
                print("    Warning: Incorrect predictions have higher certainty than correct ones")
                print("      This may indicate model overconfidence or need for calibration")

        print("\n  Feature Importance Analysis:")
        print("    Number of features with >1% importance: {}/{}".format(
            n_important_features, len(feature_importance)))
        print("    Top 5 features account for {:.1%} of total importance".format(top_5_importance))

        top_3_idx = np.argsort(feature_importance)[::-1][:3]
        print("    Top 3 features:")
        for i, idx in enumerate(top_3_idx):
            print(f"      {i + 1}. Feature {idx}: {feature_importance[idx]:.6f}")

        print("\n  Tree Depth Analysis:")
        print("    Average tree depth: {:.2f}".format(np.mean(tree_depths)))
        print("    Depth range: {} - {}".format(np.min(tree_depths), np.max(tree_depths)))
        print("    Standard deviation: {:.2f}".format(np.std(tree_depths)))

        train_acc = result['train_accuracy']
        test_acc = result['test_accuracy']
        acc_gap = train_acc - test_acc

        print("\n  Overfitting Analysis:")
        print("    Training accuracy: {:.4f} ({:.1f}%)".format(train_acc, train_acc * 100))
        print("    Test accuracy: {:.4f} ({:.1f}%)".format(test_acc, test_acc * 100))
        print("    Accuracy gap: {:.4f} ({:.1f}%)".format(acc_gap, acc_gap * 100))

        if acc_gap > 0.2:
            print("    Warning: Large accuracy gap suggests potential overfitting")
        elif acc_gap > 0.1:
            print("    Note: Moderate accuracy gap, consider regularization")
        else:
            print("    Good: Small accuracy gap indicates good generalization")

        print("\n  Data Split:")
        print("    Split ratio: 4:1 (80% Training, 20% Testing)")
        print("    Method: Subject-based splitting to avoid data leakage")

def main_rf_analysis():
    print("=" * 80)
    print("RANDOM FOREST CLASSIFICATION ANALYSIS")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("=" * 80)

    print("\nLoading and standardizing data...")
    X, y, subject_ids, repetition_ids, class_labels = load_and_prepare_data()

    if len(X) == 0:
        print("Error: No data loaded! Please check file paths.")
        return

    n_estimators = 150
    max_depth = None

    print(f"\nUsing Random Forest with {n_estimators} trees, max_depth={max_depth}")
    print("Starting binary classification analysis...")

    class_pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3)
    ]

    pair_names = [
        "40_vs_80",
        "40_vs_F40",
        "80_vs_F80",
        "F40_vs_F80"
    ]

    print("\nWill perform the following 4 classification analyses:")
    for i, (class1, class2) in enumerate(class_pairs):
        print("  {}. {} vs {}".format(i + 1, class_labels[class1], class_labels[class2]))

    all_results = []

    for (class1, class2), pair_name in zip(class_pairs, pair_names):
        print("\n" + "=" * 80)
        print("PROCESSING: {} vs {}".format(class_labels[class1], class_labels[class2]))
        print("=" * 80)

        results = evaluate_two_class_classification_rf(
            class1, class2, X, y, subject_ids, class_labels,
            n_estimators=n_estimators, max_depth=max_depth
        )

        all_results.append(results)

        feature_importance_filename = f"rf_feature_importance_{class_labels[class1]}_vs_{class_labels[class2]}_4to1_trees{n_estimators}"
        plot_rf_feature_importance(results, n_features_to_plot=20, base_filename=feature_importance_filename)

        proba_filename = f"rf_probability_{class_labels[class1]}_vs_{class_labels[class2]}_4to1_trees{n_estimators}"
        plot_rf_probability_distribution(results,
                                         [class_labels[class1], class_labels[class2]],
                                         proba_filename)

    save_fold_results_to_csv_rf(all_results, n_estimators)

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE RANDOM FOREST ANALYSIS (4:1 Split)")
    print("=" * 80)
    plot_rf_trees_analysis(all_results, base_filename="rf_trees_analysis_4to1")

    plot_comprehensive_results_rf(all_results, n_estimators, base_filename=f"rf_comprehensive_results_4to1_trees{n_estimators}")

    print("\n" + "=" * 100)
    print("OVERALL RESULTS SUMMARY - RANDOM FOREST ({} trees)".format(n_estimators))
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 100)
    print("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<25}".format(
        'Classification Task', 'Train Acc', 'Train F1', 'Test Acc', 'Test F1', 'ROC AUC', 'CV Acc (±std)'))
    print("-" * 120)

    for result in all_results:
        print("{:<25} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:.4f} (±{:.4f})".format(
            result['class_pair'],
            result['train_accuracy'],
            result['train_f1'],
            result['test_accuracy'],
            result['test_f1'],
            result['roc_auc'],
            result['cv_mean'],
            result['cv_std']))

    print("\n" + "=" * 100)
    print("CROSS-VALIDATION DETAILS SUMMARY (4:1 Split)")
    print("=" * 100)
    print("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        'Classification Task', 'CV Fold 1', 'CV Fold 2', 'CV Fold 3', 'CV Fold 4', 'CV Fold 5', 'CV F1 (Mean)'))
    print("-" * 120)

    for result in all_results:
        cv_folds_details = result.get('cv_folds_details', [])
        cv_accuracies = [fold['accuracy'] for fold in cv_folds_details]
        cv_f1_scores = [fold['f1_score'] for fold in cv_folds_details]

        cv_accuracies_display = cv_accuracies + [np.nan] * (5 - len(cv_accuracies))
        cv_f1_mean = np.mean(cv_f1_scores) if cv_f1_scores else np.nan

        def format_acc(acc):
            if isinstance(acc, (int, float)) and not np.isnan(acc):
                return f"{acc:.4f}"
            else:
                return "N/A"

        print("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15.4f}".format(
            result['class_pair'],
            format_acc(cv_accuracies_display[0]),
            format_acc(cv_accuracies_display[1]),
            format_acc(cv_accuracies_display[2]),
            format_acc(cv_accuracies_display[3]),
            format_acc(cv_accuracies_display[4]),
            cv_f1_mean
        ))

    detailed_results = []
    for result in all_results:
        cm = result['confusion_matrix']
        report = result['class_report']

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        feature_importance = result['feature_importances']
        top_feature_idx = np.argmax(feature_importance)
        top_feature_importance = feature_importance[top_feature_idx]

        tree_depths = result['tree_depths']
        avg_tree_depth = np.mean(tree_depths)
        max_tree_depth = np.max(tree_depths)

        cv_folds_details = result.get('cv_folds_details', [])
        cv_accuracies = [fold['accuracy'] for fold in cv_folds_details]
        cv_f1_scores = [fold['f1_score'] for fold in cv_folds_details]

        detailed_result = {
            'class_pair': result['class_pair'],
            'n_estimators': n_estimators,
            'max_depth': 'None' if result['max_depth'] is None else result['max_depth'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'roc_auc': result['roc_auc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std'],
            'cv_fold1_acc': cv_accuracies[0] if len(cv_accuracies) > 0 else np.nan,
            'cv_fold2_acc': cv_accuracies[1] if len(cv_accuracies) > 1 else np.nan,
            'cv_fold3_acc': cv_accuracies[2] if len(cv_accuracies) > 2 else np.nan,
            'cv_fold4_acc': cv_accuracies[3] if len(cv_accuracies) > 3 else np.nan,
            'cv_fold5_acc': cv_accuracies[4] if len(cv_accuracies) > 4 else np.nan,
            'cv_fold1_f1': cv_f1_scores[0] if len(cv_f1_scores) > 0 else np.nan,
            'cv_fold2_f1': cv_f1_scores[1] if len(cv_f1_scores) > 1 else np.nan,
            'cv_fold3_f1': cv_f1_scores[2] if len(cv_f1_scores) > 2 else np.nan,
            'cv_fold4_f1': cv_f1_scores[3] if len(cv_f1_scores) > 3 else np.nan,
            'cv_fold5_f1': cv_f1_scores[4] if len(cv_f1_scores) > 4 else np.nan,
            'top_feature_idx': top_feature_idx,
            'top_feature_importance': top_feature_importance,
            'avg_feature_importance': np.mean(feature_importance),
            'std_feature_importance': np.std(feature_importance),
            'avg_tree_depth': avg_tree_depth,
            'max_tree_depth': max_tree_depth,
            'min_tree_depth': np.min(tree_depths),
            'oob_score': result.get('oob_score', 'N/A'),
            'confusion_matrix_00': cm[0, 0] if cm.shape[0] > 0 else 0,
            'confusion_matrix_01': cm[0, 1] if cm.shape[0] > 0 else 0,
            'confusion_matrix_10': cm[1, 0] if cm.shape[1] > 0 else 0,
            'confusion_matrix_11': cm[1, 1] if cm.shape[1] > 1 else 0,
            'confusion_matrix_00_pct': cm_percent[0, 0] if cm_percent.shape[0] > 0 else 0,
            'confusion_matrix_01_pct': cm_percent[0, 1] if cm_percent.shape[0] > 0 else 0,
            'confusion_matrix_10_pct': cm_percent[1, 0] if cm_percent.shape[1] > 0 else 0,
            'confusion_matrix_11_pct': cm_percent[1, 1] if cm_percent.shape[1] > 1 else 0,
            'class0_precision': report.get('0', {}).get('precision', 0),
            'class0_recall': report.get('0', {}).get('recall', 0),
            'class0_f1_score': report.get('0', {}).get('f1-score', 0),
            'class1_precision': report.get('1', {}).get('precision', 0),
            'class1_recall': report.get('1', {}).get('recall', 0),
            'class1_f1_score': report.get('1', {}).get('f1-score', 0),
            'accuracy': report.get('accuracy', 0),
            'macro_avg_precision': report.get('macro avg', {}).get('precision', 0),
            'macro_avg_recall': report.get('macro avg', {}).get('recall', 0),
            'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0),
            'weighted_avg_precision': report.get('weighted avg', {}).get('precision', 0),
            'weighted_avg_recall': report.get('weighted avg', {}).get('recall', 0),
            'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0),
            'standardization_method': result.get('standardization_method', 'Z-score (StandardScaler)'),
            'data_split': '4:1 Subject-based',
            'split_ratio': '80% Training, 20% Testing'
        }
        detailed_results.append(detailed_result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(output_root, f'RF_detailed_results_4to1_trees{n_estimators}.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed Random Forest statistical results saved to: {detailed_csv_path}")

    simple_df = detailed_df[['class_pair', 'n_estimators', 'train_accuracy', 'train_f1',
                             'test_accuracy', 'test_f1', 'roc_auc', 'cv_mean', 'cv_std',
                             'cv_f1_mean', 'cv_f1_std', 'avg_tree_depth', 'top_feature_importance',
                             'standardization_method', 'data_split']]
    summary_csv_path = os.path.join(output_root, f'RF_summary_results_4to1_trees{n_estimators}.csv')
    simple_df.to_csv(summary_csv_path, index=False)
    print(f"Concise Random Forest results saved to: {summary_csv_path}")

    analyze_rf_performance(all_results, n_estimators)

    return all_results, detailed_df

if __name__ == "__main__":
    print("=" * 80)
    print("RANDOM FOREST CLASSIFICATION ANALYSIS")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("Same subject data not in both train and test sets")
    print("=" * 80)

    all_results, detailed_df = main_rf_analysis()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_root}/")
    print("   - PNG format (high resolution: 300 DPI)")
    print("   - SVG format (vector graphics, scalable)")
    print("   - PDF format (vector graphics)")
    print("\nGenerated files include:")
    print("   1. rf_confusion_matrix_*_4to1 - Confusion matrices (percent only)")
    print("   2. rf_cv_confusion_matrices_*_4to1 - Cross-validation confusion matrices")
    print("   3. rf_avg_cv_confusion_matrix_*_4to1 - Average cross-validation confusion matrix")
    print("   4. rf_feature_importance_*_4to1 - Feature importance plots")
    print("   5. rf_probability_*_4to1 - Probability distribution plots")
    print("   6. rf_trees_analysis_4to1 - Trees analysis plots")
    print("   7. rf_comprehensive_results_4to1_* - Comprehensive results summary")
    print("   8. RF_detailed_results_4to1_*.csv - Detailed statistical results")
    print("   9. RF_summary_results_4to1_*.csv - Concise summary results")
    print("  10. RF_all_folds_detailed_4to1_*.csv - All folds detailed results")
    print("  11. RF_overall_summary_4to1_*.csv - Overall summary with fold details")
    print("  12. RF_fold_results_*_4to1_*.csv - Per-task fold results")