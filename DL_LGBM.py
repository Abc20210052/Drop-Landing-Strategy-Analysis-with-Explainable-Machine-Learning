import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import json

warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

output_root = 'LightGBM_Analysis_Results_4to1'
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"创建主输出目录: {output_root}")

file_paths = [
    r'F:\DL_40.csv',
    r'F:\DL_80.csv',
    r'F:\DL_P40.csv',
    r'F:\DL_P80.csv'
]

class_labels = ['DL_40', 'DL_80', 'DL_P40', 'DL_P80']

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

    print("\n✓ Z-score standardization completed successfully!")
    print("✓ All features scaled to mean=0, std=1")

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

def save_figure(fig, filename, dpi=300):
    os.makedirs(output_root, exist_ok=True)
    base_path = os.path.join(output_root, filename)

    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存PNG: {png_path}")

    svg_path = f"{base_path}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存SVG: {svg_path}")

    pdf_path = f"{base_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存PDF: {pdf_path}")

def plot_confusion_matrix_lgbm(cm, class_names, title, base_filename):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot_matrix = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                annotation = "{:.1f}%\n({})".format(cm_percent[i, j], cm[i, j])
            else:
                annotation = "{:.1f}%\n(0)".format(cm_percent[i, j])
            row.append(annotation)
        annot_matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Greens',
                cbar=True, square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 10, 'va': 'center'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    ax.text(0.5, -0.15, 'Overall Accuracy: {:.2%}'.format(accuracy),
            ha='center', va='center', transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))

    ax.text(0.5, -0.25, 'Total Test Samples: {}'.format(np.sum(cm)),
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10)

    plt.tight_layout()
    save_figure(fig, base_filename)
    plt.show()

def plot_feature_importance_lgbm(importance_df, title, base_filename):
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n).copy()
    top_features = top_features.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = plt.barh(range(len(top_features)), top_features['Importance'],
                    color='green', alpha=0.7, edgecolor='black')

    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + max(top_features['Importance']) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{importance:.2f}',
                 va='center', ha='left', fontsize=9)

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    save_figure(fig, base_filename)
    plt.show()

def save_fold_results_to_csv_lgbm(all_results):
    print("\n" + "=" * 60)
    print("Saving Cross-Validation Fold Results to CSV (LightGBM)")
    print("=" * 60)

    fold_summary_rows = []

    for result in all_results:
        class_pair = result['class_pair']
        class1_name = result['class1_name']
        class2_name = result['class2_name']

        cv_folds_details = result.get('cv_folds_details', [])
        for fold_info in cv_folds_details:
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
                'Train_Accuracy': fold_info.get('train_accuracy', None),
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_summary_rows.append(row)

    all_folds_df = pd.DataFrame(fold_summary_rows)
    all_folds_csv_path = os.path.join(output_root, 'LGBM_all_folds_detailed_4to1.csv')
    all_folds_df.to_csv(all_folds_csv_path, index=False)
    print(f"  Saved all folds detailed results to: {all_folds_csv_path}")

    summary_rows = []
    for result in all_results:
        row = {
            'Classification_Task': result['class_pair'],
            'Class1': result['class1_name'],
            'Class2': result['class2_name'],
            'Train_Accuracy': result['train_accuracy'],
            'Train_F1': result['train_f1'],
            'Test_Accuracy': result['test_accuracy'],
            'Test_F1': result['test_f1'],
            'ROC_AUC': result['roc_auc'],
            'CV_Mean_Accuracy': result['cv_mean'],
            'CV_Std_Accuracy': result['cv_std'],
            'CV_Mean_F1': result['cv_f1_mean'],
            'CV_Std_F1': result['cv_f1_std'],
            'CV_Train_Mean': result.get('cv_train_mean', None),
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
            'Training_Time_s': result['training_time'],
            'Num_Trees': result['num_trees'],
            'Best_Iteration': result['best_iteration'],
            'Num_Leaves': result['params'].get('num_leaves', 15),
            'Learning_Rate': result['params'].get('learning_rate', 0.01),
            'Standardization': result.get('standardization_method', 'Z-score'),
            'Data_Split': result['data_split']
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_root, 'LGBM_overall_summary_4to1.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved overall summary to: {summary_csv_path}")

    for result in all_results:
        class_pair = result['class_pair']
        cv_folds_details = result.get('cv_folds_details', [])
        fold_rows = []
        for fold_info in cv_folds_details:
            row = {
                'Fold': fold_info['fold'],
                'Train_Subjects': fold_info['train_subjects'],
                'Val_Subjects': fold_info['val_subjects'],
                'Train_Samples': fold_info['train_samples'],
                'Val_Samples': fold_info['val_samples'],
                'Accuracy': fold_info['accuracy'],
                'F1_Score': fold_info['f1_score'],
                'Train_Accuracy': fold_info.get('train_accuracy', None),
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        filename = f'LGBM_fold_results_{class_pair}_4to1.csv'
        fold_csv_path = os.path.join(output_root, filename)
        fold_df.to_csv(fold_csv_path, index=False)
        print(f"  Saved fold results for {class_pair} to: {fold_csv_path}")

    print("\n✓ All cross-validation fold results saved to CSV files")

def evaluate_two_class_classification_lgbm(class1, class2, X, y, subject_ids, class_labels, params=None, num_boost_round=200):
    print("\n" + "=" * 60)
    print("LightGBM Classification: {} vs {}".format(class_labels[class1], class_labels[class2]))
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
    print("✓ Data was already standardized during loading (Z-score)")
    print("✓ Features have mean ≈ 0 and std ≈ 1")

    X_train_scaled = X_train
    X_test_scaled = X_test

    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }

    print("\nLightGBM Parameters:")
    for key, value in params.items():
        if key not in ['verbose', 'random_state']:
            print(f"  {key}: {value}")
    print(f"  num_boost_round: {num_boost_round}")

    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    print("\nTraining LightGBM model...")
    start_time = time.time()

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    y_train_pred_proba = model.predict(X_train_scaled)
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='binary')

    y_test_pred_proba = model.predict(X_test_scaled)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='binary')

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    else:
        roc_auc = 0.0

    unique_subjects_cv = np.unique(subject_ids_binary)
    subject_labels_list = []

    for subj in unique_subjects_cv:
        subj_indices = np.where(subject_ids_binary == subj)[0]
        subj_label = y_binary[subj_indices[0]]
        subject_labels_list.append(subj_label)

    cv_scores = []
    cv_f1_scores = []
    cv_train_scores = []
    cv_folds_details = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_subjects = []
    for subj in unique_subjects_cv:
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
        train_subjects = unique_subjects_cv[train_idx]
        val_subjects = unique_subjects_cv[val_idx]

        train_mask = np.isin(subject_ids_binary, train_subjects)
        val_mask = np.isin(subject_ids_binary, val_subjects)

        X_cv_train, X_cv_val = X_binary[train_mask], X_binary[val_mask]
        y_cv_train, y_cv_val = y_binary[train_mask], y_binary[val_mask]

        train_data_cv = lgb.Dataset(X_cv_train, label=y_cv_train)
        val_data_cv = lgb.Dataset(X_cv_val, label=y_cv_val, reference=train_data_cv)

        model_cv = lgb.train(
            params,
            train_data_cv,
            num_boost_round=num_boost_round,
            valid_sets=[val_data_cv],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )

        y_cv_pred_proba = model_cv.predict(X_cv_val)
        y_cv_pred = (y_cv_pred_proba >= 0.5).astype(int)
        cv_score = accuracy_score(y_cv_val, y_cv_pred)
        cv_f1 = f1_score(y_cv_val, y_cv_pred, average='binary')

        y_cv_train_pred_proba = model_cv.predict(X_cv_train)
        y_cv_train_pred = (y_cv_train_pred_proba >= 0.5).astype(int)
        cv_train_score = accuracy_score(y_cv_train, y_cv_train_pred)

        cv_scores.append(cv_score)
        cv_f1_scores.append(cv_f1)
        cv_train_scores.append(cv_train_score)

        cv_cm = confusion_matrix(y_cv_val, y_cv_pred)

        cv_folds_details.append({
            'fold': fold,
            'accuracy': cv_score,
            'f1_score': cv_f1,
            'train_accuracy': cv_train_score,
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
        print("    Val Accuracy = {:.4f}, Val F1 = {:.4f}, Train Accuracy = {:.4f}".format(
            cv_score, cv_f1, cv_train_score))
        print("    Confusion Matrix:")
        print("      [[{}  {}]".format(cv_cm[0, 0], cv_cm[0, 1] if cv_cm.shape[1] > 1 else 0))
        print("       [{}  {}]]".format(cv_cm[1, 0] if cv_cm.shape[0] > 1 else 0,
                                        cv_cm[1, 1] if cv_cm.shape[0] > 1 and cv_cm.shape[1] > 1 else 0))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_f1_mean = np.mean(cv_f1_scores)
    cv_f1_std = np.std(cv_f1_scores)
    cv_train_mean = np.mean(cv_train_scores)

    print("-" * 60)
    print("Cross-Validation Summary:")
    print("  Mean Val Accuracy = {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("  Mean Val F1 Score = {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))
    print("  Mean Train Accuracy = {:.4f}".format(cv_train_mean))
    print("  Individual Val Accuracy scores: {}".format([round(s, 4) for s in cv_scores]))
    print("  Individual Val F1 scores: {}".format([round(s, 4) for s in cv_f1_scores]))
    print("-" * 60)

    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("Results Summary ({} vs {})".format(class_labels[class1], class_labels[class2]))
    print("=" * 60)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    print("Training F1 Score: {:.4f}".format(train_f1))
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Test F1 Score: {:.4f}".format(test_f1))
    print("ROC AUC Score: {:.4f}".format(roc_auc))
    print("Cross-validation Mean Val Accuracy: {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("Cross-validation Mean Val F1 Score: {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))
    print("Cross-validation Mean Train Accuracy: {:.4f}".format(cv_train_mean))

    print("\nLightGBM Model Information:")
    print("  Number of trees: {}".format(model.num_trees()))
    print("  Number of features: {}".format(model.num_feature()))
    print("  Best iteration: {}".format(model.best_iteration))

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=[class_labels[class1], class_labels[class2]]))

    print("Confusion Matrix (Counts):")
    print(cm)

    plot_title = "LightGBM: {} vs {}\n(4:1 Split)".format(
        class_labels[class1], class_labels[class2])
    cm_filename = f"lgbm_confusion_matrix_{class_labels[class1]}_vs_{class_labels[class2]}_4to1"
    plot_confusion_matrix_lgbm(cm,
                               [class_labels[class1], class_labels[class2]],
                               plot_title,
                               cm_filename)

    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Feature Importance:")
    print(importance_df.head(10).to_string(index=False))

    fi_title = f"LightGBM Feature Importance: {class_labels[class1]} vs {class_labels[class2]} (4:1 Split)"
    fi_filename = f"lgbm_feature_importance_{class_labels[class1]}_vs_{class_labels[class2]}_4to1"
    plot_feature_importance_lgbm(importance_df, fi_title, fi_filename)

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
        'cv_train_mean': cv_train_mean,
        'cv_scores': cv_scores,
        'cv_f1_scores': cv_f1_scores,
        'cv_folds_details': cv_folds_details,
        'confusion_matrix': cm,
        'class_report': classification_report(y_test, y_test_pred,
                                              target_names=[class_labels[class1], class_labels[class2]],
                                              output_dict=True),
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_pred_proba,
        'model': model,
        'params': params,
        'training_time': training_time,
        'num_trees': model.num_trees(),
        'best_iteration': model.best_iteration,
        'feature_importance': importance_df,
        'standardization_method': 'Z-score (StandardScaler)',
        'data_split': '4:1 Subject-based',
        'split_ratio': '80% Training, 20% Testing'
    }

    return results

def plot_lgbm_probability_analysis(results, class_labels_pair, base_filename):
    y_true = results['y_true']
    y_proba = results['y_proba']
    y_pred = results['y_pred']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for class_idx, class_name in enumerate(class_labels_pair):
        class_mask = (y_true == class_idx)
        class_proba = y_proba[class_mask]

        axes[0, 0].hist(class_proba, bins=20, alpha=0.6, label=f'True {class_name}',
                        density=True, edgecolor='black')

    axes[0, 0].set_xlabel('Predicted Probability for Class 1', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Probability Distribution by True Class (4:1 Split)\n(Higher = more confident for Class 1)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])

    epsilon = 1e-10
    p = y_proba
    entropy = -(p * np.log(p + epsilon) + (1 - p) * np.log(1 - p + epsilon))

    axes[0, 1].hist(entropy, bins=20, alpha=0.7, color='green',
                    density=True, edgecolor='black')

    avg_entropy = np.mean(entropy)
    axes[0, 1].axvline(x=avg_entropy, color='red', linestyle='--', linewidth=2,
                       label=f'Avg Entropy: {avg_entropy:.3f}')

    axes[0, 1].set_xlabel('Prediction Entropy', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Prediction Uncertainty (Entropy) (4:1 Split)\n(Higher = more uncertain)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    max_proba = np.maximum(y_proba, 1 - y_proba)
    correct_mask = (y_true == y_pred)

    proba_bins = np.linspace(0.5, 1.0, 11)
    accuracy_by_bin = []
    samples_by_bin = []

    for i in range(len(proba_bins) - 1):
        bin_mask = (max_proba >= proba_bins[i]) & (max_proba < proba_bins[i + 1])
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.sum(correct_mask[bin_mask]) / np.sum(bin_mask)
            accuracy_by_bin.append(bin_accuracy)
            samples_by_bin.append(np.sum(bin_mask))
        else:
            accuracy_by_bin.append(0)
            samples_by_bin.append(0)

    bin_centers = (proba_bins[:-1] + proba_bins[1:]) / 2
    axes[1, 0].plot(bin_centers, accuracy_by_bin, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].fill_between(bin_centers, accuracy_by_bin, alpha=0.3, color='green')

    for i, (center, acc, n_samples) in enumerate(zip(bin_centers, accuracy_by_bin, samples_by_bin)):
        if n_samples > 0:
            axes[1, 0].text(center, acc + 0.02, f'n={n_samples}',
                            ha='center', va='bottom', fontsize=8)

    axes[1, 0].set_xlabel('Maximum Probability (Confidence)', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Accuracy vs. Prediction Confidence (4:1 Split)\n(Calibration Analysis)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].set_xlim([0.5, 1.0])

    correct_proba = max_proba[correct_mask]
    incorrect_proba = max_proba[~correct_mask] if np.sum(~correct_mask) > 0 else []

    data_to_plot = [correct_proba]
    labels = ['Correct']
    if len(incorrect_proba) > 0:
        data_to_plot.append(incorrect_proba)
        labels.append('Incorrect')

    bp = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)

    colors = ['lightgreen', 'lightcoral'] if len(data_to_plot) > 1 else ['lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1, 1].set_ylabel('Maximum Probability', fontsize=12)
    axes[1, 1].set_title('Prediction Confidence (4:1 Split):\nCorrect vs Incorrect Predictions',
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0.4, 1.05])

    if len(correct_proba) > 0:
        axes[1, 1].text(0.02, 0.95, f'Correct: μ={np.mean(correct_proba):.3f}, σ={np.std(correct_proba):.3f}',
                        transform=axes[1, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    if len(incorrect_proba) > 0:
        axes[1, 1].text(0.02, 0.85, f'Incorrect: μ={np.mean(incorrect_proba):.3f}, σ={np.std(incorrect_proba):.3f}',
                        transform=axes[1, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()

    if base_filename:
        save_figure(fig, base_filename)

    plt.show()

def plot_lgbm_parameter_comparison(X, y, subject_ids, class_labels, class_pairs):
    print("\n" + "=" * 60)
    print("LightGBM Parameter Comparison (4:1 Split)")
    print("=" * 60)

    parameter_sets = {
        'Default': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8
        },
        'Conservative': {
            'num_leaves': 15,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.6
        },
        'Aggressive': {
            'num_leaves': 63,
            'learning_rate': 0.1,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.9
        },
        'Balanced': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    results_by_param = {param_name: [] for param_name in parameter_sets.keys()}

    for idx, (class1, class2) in enumerate(class_pairs):
        print(f"\nAnalyzing {class_labels[class1]} vs {class_labels[class2]}:")

        mask = (y == class1) | (y == class2)
        X_binary = X[mask]
        y_binary = y[mask]
        subject_ids_binary = subject_ids[mask]

        y_binary = np.where(y_binary == class1, 0, 1)

        train_indices, test_indices = create_cross_subject_split_4_to_1(
            X_binary, y_binary, subject_ids_binary, test_size=0.2
        )

        X_train, X_test = X_binary[train_indices], X_binary[test_indices]
        y_train, y_test = y_binary[train_indices], y_binary[test_indices]

        test_accuracies = []
        test_f1_scores = []
        training_times = []

        for param_name, param_set in parameter_sets.items():
            print(f"  {param_name}: ", end='')

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
            params.update(param_set)

            start_time = time.time()

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            training_time = time.time() - start_time
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='binary')

            test_accuracies.append(test_acc)
            test_f1_scores.append(test_f1)
            training_times.append(training_time)

            results_by_param[param_name].append(test_acc)

            print(f"Test Acc={test_acc:.4f}, F1={test_f1:.4f}, Time={training_time:.2f}s, Trees={model.num_trees()}")

        x_pos = np.arange(len(parameter_sets))
        width = 0.35

        bars1 = axes[idx].bar(x_pos - width / 2, test_accuracies, width,
                              label='Test Accuracy', color='green', alpha=0.8)

        ax2 = axes[idx].twinx()
        bars2 = ax2.bar(x_pos + width / 2, training_times, width,
                        label='Training Time (s)', color='orange', alpha=0.8)

        axes[idx].set_xlabel('Parameter Set', fontsize=12)
        axes[idx].set_ylabel('Test Accuracy', fontsize=12, color='green')
        ax2.set_ylabel('Training Time (s)', fontsize=12, color='orange')
        axes[idx].set_title(f'{class_labels[class1]} vs {class_labels[class2]} (4:1 Split)',
                            fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(list(parameter_sets.keys()), rotation=45, ha='right')

        axes[idx].tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='orange')

        lines1, labels1 = axes[idx].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[idx].legend(lines1 + lines2, labels1 + labels2, loc='best')

        axes[idx].grid(True, alpha=0.3, axis='x')
        axes[idx].set_ylim([0, 1.1])

        for i, (acc, time_val) in enumerate(zip(test_accuracies, training_times)):
            axes[idx].text(i - width / 2, acc + 0.02, f'{acc:.3f}',
                           ha='center', va='bottom', fontsize=9, color='green')
            ax2.text(i + width / 2, time_val + max(training_times) * 0.01, f'{time_val:.1f}',
                     ha='center', va='bottom', fontsize=9, color='orange')

        best_idx = np.argmax(test_f1_scores)
        best_param = list(parameter_sets.keys())[best_idx]
        best_f1 = test_f1_scores[best_idx]

        axes[idx].text(0.02, 0.95, f'Best (F1): {best_param}\nF1: {best_f1:.4f}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_root, 'lgbm_parameter_comparison_4to1.png'), dpi=300, bbox_inches='tight')
    print("\nLightGBM parameter comparison plot saved to: lgbm_parameter_comparison_4to1.png")
    plt.show()

    print("\n" + "=" * 60)
    print("PARAMETER SET PERFORMANCE SUMMARY (4:1 Split)")
    print("=" * 60)
    print("{:<15} {:<10} {:<10} {:<10}".format(
        'Parameter Set', 'Mean Acc', 'Std Acc', 'Mean F1'))
    print("-" * 60)

    for param_name in parameter_sets.keys():
        accuracies = results_by_param[param_name]
        mean_acc = np.mean(accuracies) if accuracies else 0
        std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
        print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            param_name, mean_acc, std_acc, mean_acc))

def main_lgbm_analysis():
    print("=" * 80)
    print("LIGHTGBM CLASSIFICATION ANALYSIS")
    print("Gradient Boosting Decision Trees")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("=" * 80)

    print("\nLoading and standardizing data...")
    X, y, subject_ids, repetition_ids, class_labels = load_and_prepare_data()

    if len(X) == 0:
        print("Error: No data loaded! Please check file paths.")
        return

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    num_boost_round = 200

    print(f"\nUsing LightGBM with parameters:")
    for key, value in params.items():
        if key not in ['verbose', 'random_state']:
            print(f"  {key}: {value}")
    print(f"  num_boost_round: {num_boost_round}")

    print("Starting binary classification analysis...")

    class_pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3)
    ]

    pair_names = [
        "DL_40_vs_DL_80",
        "DL_40_vs_DL_P40",
        "DL_80_vs_DL_P80",
        "DL_P40_vs_DL_P80"
    ]

    print("\nWill perform the following 4 classification analyses:")
    for i, (class1, class2) in enumerate(class_pairs):
        print("  {}. {} vs {}".format(i + 1, class_labels[class1], class_labels[class2]))

    all_results = []

    for (class1, class2), pair_name in zip(class_pairs, pair_names):
        print("\n" + "=" * 80)
        print("PROCESSING: {} vs {}".format(class_labels[class1], class_labels[class2]))
        print("=" * 80)

        results = evaluate_two_class_classification_lgbm(
            class1, class2, X, y, subject_ids, class_labels,
            params=params, num_boost_round=num_boost_round
        )

        all_results.append(results)

        proba_analysis_filename = f"lgbm_probability_analysis_{class_labels[class1]}_vs_{class_labels[class2]}_4to1"
        plot_lgbm_probability_analysis(results,
                                       [class_labels[class1], class_labels[class2]],
                                       proba_analysis_filename)

    save_fold_results_to_csv_lgbm(all_results)

    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT LIGHTGBM PARAMETER SETTINGS (4:1 Split)")
    print("=" * 80)
    plot_lgbm_parameter_comparison(X, y, subject_ids, class_labels, class_pairs)

    plot_comprehensive_results_lgbm(all_results)

    print("\n" + "=" * 100)
    print("OVERALL RESULTS SUMMARY - LIGHTGBM (4:1 Split)")
    print(f"Parameters: num_leaves={params['num_leaves']}, learning_rate={params['learning_rate']}")
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 100)
    print("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<20} {:<15}".format(
        'Classification Task', 'Train Acc', 'Train F1', 'Test Acc', 'Test F1', 'ROC AUC', 'CV Acc (±std)',
        'CV F1 (±std)'))
    print("-" * 140)

    for result in all_results:
        print("{:<25} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:.4f} (±{:.4f}) {:.4f} (±{:.4f})".format(
            result['class_pair'],
            result['train_accuracy'],
            result['train_f1'],
            result['test_accuracy'],
            result['test_f1'],
            result['roc_auc'],
            result['cv_mean'],
            result['cv_std'],
            result['cv_f1_mean'],
            result['cv_f1_std']))

    detailed_results = []
    for result in all_results:
        cm = result['confusion_matrix']
        report = result['class_report']

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        detailed_result = {
            'class_pair': result['class_pair'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'roc_auc': result['roc_auc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std'],
            'cv_scores': str([round(s, 4) for s in result['cv_scores']]),
            'cv_f1_scores': str([round(s, 4) for s in result['cv_f1_scores']]),
            'cv_train_mean': result.get('cv_train_mean', 0),
            'training_time': result['training_time'],
            'num_trees': result['num_trees'],
            'best_iteration': result['best_iteration'],
            'confusion_matrix_00': cm[0, 0] if cm.shape[0] > 0 else 0,
            'confusion_matrix_01': cm[0, 1] if cm.shape[0] > 0 else 0,
            'confusion_matrix_10': cm[1, 0] if cm.shape[1] > 0 else 0,
            'confusion_matrix_11': cm[1, 1] if cm.shape[1] > 0 else 0,
            'confusion_matrix_00_pct': cm_percent[0, 0] if cm_percent.shape[0] > 0 else 0,
            'confusion_matrix_01_pct': cm_percent[0, 1] if cm_percent.shape[0] > 0 else 0,
            'confusion_matrix_10_pct': cm_percent[1, 0] if cm_percent.shape[1] > 0 else 0,
            'confusion_matrix_11_pct': cm_percent[1, 1] if cm_percent.shape[1] > 0 else 0,
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
            'split_ratio': '80% Training, 20% Testing',
            'num_leaves': result['params'].get('num_leaves', 15),
            'learning_rate': result['params'].get('learning_rate', 0.01)
        }
        detailed_results.append(detailed_result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(output_root, 'LGBM_detailed_results_4to1.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed LightGBM statistical results saved to: {detailed_csv_path}")

    simple_df = detailed_df[['class_pair', 'train_accuracy', 'train_f1', 'test_accuracy', 'test_f1',
                             'roc_auc', 'cv_mean', 'cv_std', 'cv_f1_mean', 'cv_f1_std',
                             'training_time', 'num_trees', 'best_iteration',
                             'standardization_method', 'data_split']]
    summary_csv_path = os.path.join(output_root, 'LGBM_summary_results_4to1.csv')
    simple_df.to_csv(summary_csv_path, index=False)
    print(f"Concise LightGBM results saved to: {summary_csv_path}")

    analyze_lgbm_performance(all_results)

    save_lgbm_models(all_results)

    return all_results, detailed_df

def plot_comprehensive_results_lgbm(all_results):
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
    training_times = [result['training_time'] for result in all_results]
    num_trees_list = [result['num_trees'] for result in all_results]

    x = np.arange(len(class_pairs))
    width = 0.25

    axes[0].bar(x - width, train_accuracies, width, label='Training Accuracy', color='lightgreen', alpha=0.8,
                edgecolor='black')
    axes[0].bar(x, test_accuracies, width, label='Test Accuracy', color='green', alpha=0.8, edgecolor='black')
    axes[0].bar(x + width, cv_means, width, label='CV Accuracy', color='darkgreen', alpha=0.8,
                yerr=cv_stds, capsize=5, edgecolor='black')

    axes[0].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('LightGBM Accuracy Comparison (4:1 Split)\n(Z-score Standardized)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_pairs, rotation=45, ha='right')
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

    axes[1].bar(x - width, train_f1_scores, width, label='Training F1', color='#90EE90', alpha=0.8, edgecolor='black')
    axes[1].bar(x, test_f1_scores, width, label='Test F1', color='#32CD32', alpha=0.8, edgecolor='black')
    axes[1].bar(x + width, cv_f1_means, width, label='CV F1', color='#006400', alpha=0.8,
                yerr=cv_f1_stds, capsize=5, edgecolor='black')

    axes[1].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('LightGBM F1 Score Comparison (4:1 Split)\n(Z-score Standardized)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_pairs, rotation=45, ha='right')
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

    time_bars = axes[2].bar(x_pos - width / 2, training_times, width,
                            label='Training Time (s)', color='lightgreen', alpha=0.8, edgecolor='black')

    ax2 = axes[2].twinx()
    tree_bars = ax2.bar(x_pos + width / 2, num_trees_list, width,
                        label='Number of Trees', color='darkgreen', alpha=0.8, edgecolor='black')

    axes[2].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Training Time (s)', fontsize=12, fontweight='bold', color='green')
    ax2.set_ylabel('Number of Trees', fontsize=12, fontweight='bold', color='darkgreen')
    axes[2].set_title('Training Performance Analysis (4:1 Split)\n(Time and Model Complexity)',
                      fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_pairs, rotation=45, ha='right')

    axes[2].tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    axes[2].grid(True, alpha=0.3, axis='x')

    for i, (time_val, n_trees) in enumerate(zip(training_times, num_trees_list)):
        axes[2].text(i - width / 2, time_val + max(training_times) * 0.01, f'{time_val:.1f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
        ax2.text(i + width / 2, n_trees + max(num_trees_list) * 0.01, f'{n_trees}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')

    accuracy_gaps = [train - test for train, test in zip(train_accuracies, test_accuracies)]

    x_pos = np.arange(len(class_pairs))
    bars = axes[3].bar(x_pos, accuracy_gaps,
                       color=['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' for gap in accuracy_gaps],
                       alpha=0.8, edgecolor='black')

    axes[3].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate overfitting threshold')
    axes[3].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High overfitting threshold')
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    axes[3].set_xlabel('Classification Task', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Accuracy Gap (Train - Test)', fontsize=12, fontweight='bold')
    axes[3].set_title('LightGBM Overfitting Analysis (4:1 Split)\n(Large gaps indicate overfitting)',
                      fontsize=14, fontweight='bold')
    axes[3].set_xticks(x_pos)
    axes[3].set_xticklabels(class_pairs, rotation=45, ha='right')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3, axis='y')

    for bar, gap, train_acc, test_acc in zip(bars, accuracy_gaps, train_accuracies, test_accuracies):
        height = bar.get_height()
        y_pos = height + 0.01 if height >= 0 else height - 0.03
        axes[3].text(bar.get_x() + bar.get_width() / 2., y_pos,
                     f'{gap:.3f}\n({train_acc:.3f}-{test_acc:.3f})',
                     ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_root, 'lgbm_comprehensive_results_4to1.png'), dpi=300, bbox_inches='tight')
    print("\nComprehensive LightGBM results plot saved to: lgbm_comprehensive_results_4to1.png")
    plt.show()

def analyze_lgbm_performance(all_results):
    print("\n" + "=" * 80)
    print("LIGHTGBM CLASSIFIER SPECIFIC ANALYSIS (4:1 Split)")
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 80)

    for result in all_results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_proba = result['y_proba']
        training_time = result['training_time']
        num_trees = result['num_trees']
        best_iteration = result['best_iteration']
        feature_importance = result['feature_importance']

        correct_mask = (y_true == y_pred)

        max_proba = np.maximum(y_proba, 1 - y_proba)
        avg_certainty_correct = np.mean(max_proba[correct_mask]) if np.sum(correct_mask) > 0 else 0
        avg_certainty_incorrect = np.mean(max_proba[~correct_mask]) if np.sum(~correct_mask) > 0 else 0

        epsilon = 1e-10
        p = y_proba
        entropy = -(p * np.log(p + epsilon) + (1 - p) * np.log(1 - p + epsilon))
        avg_entropy = np.mean(entropy)

        print("\n{}:".format(result['class_pair']))
        print("  F1 Scores:")
        print("    Training F1: {:.4f}".format(result['train_f1']))
        print("    Test F1: {:.4f}".format(result['test_f1']))
        print("    CV F1: {:.4f} (±{:.4f})".format(result['cv_f1_mean'], result['cv_f1_std']))

        print("\n  Model Information:")
        print("    Number of trees: {}".format(num_trees))
        print("    Best iteration: {}".format(best_iteration))
        print("    Training time: {:.2f} seconds".format(training_time))
        print("    Number of features used: {}".format(len(feature_importance)))

        print("\n  Feature Importance Analysis:")
        top_features = feature_importance.head(5)
        for idx, row in top_features.iterrows():
            print("    {}: {:.4f}".format(row['Feature'], row['Importance']))

        print("\n  Prediction Analysis:")
        print("    Correct predictions: {}/{} ({:.1%})".format(
            np.sum(correct_mask), len(y_true), np.sum(correct_mask) / len(y_true)))
        print("    Average prediction entropy: {:.4f}".format(avg_entropy))
        print("    Certainty analysis:")
        print("      Correct predictions - Avg certainty: {:.4f}".format(avg_certainty_correct))
        print("      Incorrect predictions - Avg certainty: {:.4f}".format(avg_certainty_incorrect))

        if avg_certainty_incorrect > 0 and avg_certainty_correct > 0:
            certainty_ratio = avg_certainty_incorrect / avg_certainty_correct
            if certainty_ratio > 1.0:
                print("      ⚠️  Warning: Incorrect predictions have higher certainty than correct ones")

        train_acc = result['train_accuracy']
        test_acc = result['test_accuracy']
        cv_train_mean = result.get('cv_train_mean', 0)
        acc_gap = train_acc - test_acc
        cv_gap = cv_train_mean - result['cv_mean']

        print("\n  Overfitting Analysis:")
        print("    Training accuracy: {:.4f}".format(train_acc))
        print("    Test accuracy: {:.4f}".format(test_acc))
        print("    Accuracy gap: {:.4f}".format(acc_gap))
        print("    CV train-test gap: {:.4f}".format(cv_gap))

        if acc_gap > 0.2:
            print("    ⚠️  HIGH OVERFITTING: Large accuracy gap (>0.2)")
            print("      Suggestions: Increase feature_fraction, reduce num_leaves, increase min_child_samples")
        elif acc_gap > 0.15:
            print("    ⚠️  Moderate overfitting: Accuracy gap > 0.15")
            print("      Consider: Reduce learning_rate, increase bagging_fraction")
        elif acc_gap > 0.1:
            print("    ⚠️  Slight overfitting: Accuracy gap > 0.1")
            print("      Acceptable for gradient boosting models")
        else:
            print("    ✓ Good: Small accuracy gap indicates good generalization")

        print("\n  Model Efficiency:")
        acc_per_second = test_acc / training_time if training_time > 0 else 0
        acc_per_tree = test_acc / num_trees if num_trees > 0 else 0
        print("    Accuracy per second: {:.6f}".format(acc_per_second))
        print("    Accuracy per tree: {:.6f}".format(acc_per_tree))

        roc_auc = result['roc_auc']
        print("\n  ROC AUC Analysis:")
        print("    ROC AUC: {:.4f}".format(roc_auc))

        if roc_auc > 0.9:
            print("    ✓ Excellent discrimination ability")
        elif roc_auc > 0.8:
            print("    ✓ Good discrimination ability")
        elif roc_auc > 0.7:
            print("    ⚠️  Fair discrimination ability")
        else:
            print("    ⚠️  Poor discrimination ability - consider feature engineering or different model")

        print("\n  Data Split:")
        print("    Split ratio: 4:1 (80% Training, 20% Testing)")
        print("    Method: Subject-based splitting to avoid data leakage")

def save_lgbm_models(all_results):
    print("\n" + "=" * 60)
    print("SAVING LIGHTGBM MODELS (4:1 Split)")
    print("=" * 60)

    models_dir = os.path.join(output_root, "lgbm_models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for result in all_results:
        model = result['model']
        class_pair = result['class_pair']

        model_path = os.path.join(models_dir, f"lgbm_model_{class_pair}_4to1.txt")
        model.save_model(model_path)

        params_path = os.path.join(models_dir, f"lgbm_params_{class_pair}_4to1.json")
        with open(params_path, 'w') as f:
            json.dump(result['params'], f, indent=4)

        importance_path = os.path.join(models_dir, f"lgbm_feature_importance_{class_pair}_4to1.csv")
        result['feature_importance'].to_csv(importance_path, index=False)

        print(f"  Saved {class_pair}: model, params, and feature importance")

    print(f"\nAll models saved to directory: {models_dir}")

if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM CLASSIFICATION ANALYSIS")
    print("Gradient Boosting Decision Trees")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("Same subject data not in both train and test sets")
    print("=" * 80)

    all_results, detailed_df = main_lgbm_analysis()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_root}/")
    print("   - PNG format (high resolution: 300 DPI)")
    print("   - SVG format (vector graphics, scalable)")
    print("   - PDF format (vector graphics)")
    print("\nGenerated files include:")
    print("   1. lgbm_confusion_matrix_*_4to1 - Confusion matrices")
    print("   2. lgbm_feature_importance_*_4to1 - Feature importance plots")
    print("   3. lgbm_probability_analysis_*_4to1 - Probability analysis plots")
    print("   4. lgbm_parameter_comparison_4to1 - Parameter comparison plot")
    print("   5. lgbm_comprehensive_results_4to1 - Comprehensive results summary")
    print("   6. LGBM_detailed_results_4to1.csv - Detailed statistical results")
    print("   7. LGBM_summary_results_4to1.csv - Concise summary results")
    print("   8. LGBM_all_folds_detailed_4to1.csv - All folds detailed results")
    print("   9. LGBM_overall_summary_4to1.csv - Overall summary with fold details")
    print("  10. LGBM_fold_results_*_4to1.csv - Per-task fold results")
    print("  11. lgbm_models/ - Saved models directory")