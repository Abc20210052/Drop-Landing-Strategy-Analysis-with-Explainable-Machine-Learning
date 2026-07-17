import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

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

    for i, file_path in enumerate(file_paths):
        if os.path.exists(file_path):
            print("Loading: {}".format(file_path))
            df = pd.read_csv(file_path)
            print("  Data shape: {}".format(df.shape))

            for j in range(len(df)):
                subject_id = "Class{}_Subject{}".format(i, j // 3)
                all_data.append(df.iloc[j].values)
                all_labels.append(i)
                all_subject_ids.append(subject_id)
        else:
            print("File not found: {}".format(file_path))

    X = np.array(all_data)
    y = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)

    print("\nTotal samples: {}".format(len(X)))
    print("Feature dimension: {}".format(X.shape[1]))
    print("Class distribution: {}".format(np.bincount(y)))
    print("Number of subjects: {}".format(len(np.unique(subject_ids))))

    print("\n" + "=" * 60)
    print("Performing Detailed Data Standardization")
    print("=" * 60)

    print("\n1. Original data statistics:")
    print(f"   Overall mean: {np.mean(X):.4f}")
    print(f"   Overall std: {np.std(X):.4f}")
    print(f"   Min value: {np.min(X):.4f}")
    print(f"   Max value: {np.max(X):.4f}")

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

    print("\n2. Applying Z-score standardization (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n3. Standardized data statistics:")
    print(f"   Overall mean: {np.mean(X_scaled):.6f} (should be close to 0)")
    print(f"   Overall std: {np.std(X_scaled):.6f} (should be close to 1)")
    print(f"   Min value: {np.min(X_scaled):.4f}")
    print(f"   Max value: {np.max(X_scaled):.4f}")

    print("\n4. Standardization check by group:")
    unique_groups = np.unique(y)
    for group_idx in unique_groups:
        mask = (y == group_idx)
        group_data = X_scaled[mask]
        group_name = class_labels[group_idx] if group_idx < len(class_labels) else f"Group_{group_idx}"
        print(f"   {group_name}: mean = {np.mean(group_data):.4f}, std = {np.std(group_data):.4f}")

    print("\nStandardization completed successfully!")
    print("All biomechanical variables are now on the same scale")
    print("SVM performance will be more stable and accurate")

    return X_scaled, y, subject_ids, class_labels


def create_cross_subject_split(X, y, subject_ids, test_size=0.2):
    unique_subjects = np.unique(subject_ids)

    subject_classes = {}
    for subj in unique_subjects:
        subj_indices = np.where(subject_ids == subj)[0]
        subj_class = y[subj_indices[0]]
        if subj_class not in subject_classes:
            subject_classes[subj_class] = []
        subject_classes[subj_class].append(subj)

    train_subjects = []
    test_subjects = []

    for class_id, subjects in subject_classes.items():
        n_subjects = len(subjects)
        n_test = int(np.ceil(n_subjects * test_size))
        np.random.shuffle(subjects)
        test_subjects.extend(subjects[:n_test])
        train_subjects.extend(subjects[n_test:])

    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

    print("Training subjects: {}, Training samples: {}".format(len(train_subjects), len(train_indices)))
    print("Test subjects: {}, Test samples: {}".format(len(test_subjects), len(test_indices)))

    return train_indices, test_indices


def plot_confusion_matrix(cm, class_names, title, save_path=None):
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

    sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Blues',
                cbar=True, square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 10, 'va': 'center'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to: {}".format(save_path))

    plt.show()


def evaluate_two_class_classification(class1, class2, X, y, subject_ids, class_labels):
    print("\n" + "=" * 60)
    print("Classification: {} vs {}".format(class_labels[class1], class_labels[class2]))
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

    train_indices, test_indices = create_cross_subject_split(
        X_binary, y_binary, subject_ids_binary, test_size=0.2
    )

    X_train, X_test = X_binary[train_indices], X_binary[test_indices]
    y_train, y_test = y_binary[train_indices], y_binary[test_indices]

    print("\nData Standardization Status:")
    print("Data was already standardized during loading (Z-score)")
    print("Features have mean ≈ 0 and std ≈ 1")
    print("SVM performance is optimized for standardized data")
    print("No additional standardization needed")

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    y_train_pred = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='binary')

    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='binary')

    y_test_proba = svm_model.predict_proba(X_test)
    y_test_decision = svm_model.decision_function(X_test)

    unique_subjects = np.unique(subject_ids_binary)
    subject_labels_list = []

    for subj in unique_subjects:
        subj_indices = np.where(subject_ids_binary == subj)[0]
        subj_label = y_binary[subj_indices[0]]
        subject_labels_list.append(subj_label)

    cv_scores = []
    cv_f1_scores = []
    cv_fold_details = []
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

        svm_cv = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_cv.fit(X_cv_train, y_cv_train)

        y_cv_pred = svm_cv.predict(X_cv_val)
        cv_score = accuracy_score(y_cv_val, y_cv_pred)
        cv_f1 = f1_score(y_cv_val, y_cv_pred, average='binary')

        cv_scores.append(cv_score)
        cv_f1_scores.append(cv_f1)

        cm_fold = confusion_matrix(y_cv_val, y_cv_pred)

        fold_info = {
            'fold': fold,
            'train_subjects': len(train_subjects),
            'val_subjects': len(val_subjects),
            'train_samples': np.sum(train_mask),
            'val_samples': np.sum(val_mask),
            'accuracy': cv_score,
            'f1_score': cv_f1,
            'cm_00': cm_fold[0, 0] if cm_fold.shape[0] > 0 else 0,
            'cm_01': cm_fold[0, 1] if cm_fold.shape[1] > 1 else 0,
            'cm_10': cm_fold[1, 0] if cm_fold.shape[0] > 1 else 0,
            'cm_11': cm_fold[1, 1] if cm_fold.shape[0] > 1 and cm_fold.shape[1] > 1 else 0
        }
        cv_fold_details.append(fold_info)

        print("  Fold {}: Train subjects={}, Val subjects={}, Train samples={}, Val samples={}".format(
            fold, len(train_subjects), len(val_subjects),
            np.sum(train_mask), np.sum(val_mask)))
        print("    Accuracy = {:.4f}, F1 Score = {:.4f}".format(cv_score, cv_f1))
        print("    Confusion Matrix:")
        print("      [[{}  {}]".format(cm_fold[0, 0], cm_fold[0, 1] if cm_fold.shape[1] > 1 else 0))
        print("       [{}  {}]]".format(cm_fold[1, 0] if cm_fold.shape[0] > 1 else 0,
                                        cm_fold[1, 1] if cm_fold.shape[0] > 1 and cm_fold.shape[1] > 1 else 0))

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
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    print("Training F1 Score: {:.4f}".format(train_f1))
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Test F1 Score: {:.4f}".format(test_f1))
    print("Cross-validation Mean Accuracy: {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("Cross-validation Mean F1 Score: {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))

    print("\nSVM Model Parameters:")
    print("  Kernel: {}".format(svm_model.kernel))
    print("  C parameter: {}".format(svm_model.C))
    print("  Gamma: {}".format(svm_model.gamma))
    print("  Number of support vectors: {}".format(len(svm_model.support_)))
    print("  Number of features: {}".format(svm_model.n_features_in_))

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=[class_labels[class1], class_labels[class2]]))

    print("Confusion Matrix (Counts):")
    print(cm)

    plot_title = "SVM Confusion Matrix: {} vs {}\n(Data Standardized using Z-score)".format(class_labels[class1],
                                                                                            class_labels[class2])
    save_path = "svm_confusion_matrix_{}_vs_{}.png".format(class_labels[class1], class_labels[class2])

    plot_confusion_matrix(cm,
                          [class_labels[class1], class_labels[class2]],
                          plot_title,
                          save_path)

    results = {
        'class_pair': "{}_vs_{}".format(class_labels[class1], class_labels[class2]),
        'class1_name': class_labels[class1],
        'class2_name': class_labels[class2],
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_f1_mean': cv_f1_mean,
        'cv_f1_std': cv_f1_std,
        'cv_scores': cv_scores,
        'cv_f1_scores': cv_f1_scores,
        'cv_fold_details': cv_fold_details,
        'confusion_matrix': cm,
        'class_report': classification_report(y_test, y_test_pred,
                                              target_names=[class_labels[class1], class_labels[class2]],
                                              output_dict=True),
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'y_decision': y_test_decision,
        'model': svm_model,
        'n_support_vectors': len(svm_model.support_),
        'svm_parameters': {
            'kernel': svm_model.kernel,
            'C': svm_model.C,
            'gamma': svm_model.gamma
        }
    }

    return results


def plot_svm_decision_boundary_analysis(results, class_labels_pair, save_path=None):
    y_true = results['y_true']
    y_decision = results['y_decision']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for class_idx, class_name in enumerate(class_labels_pair):
        class_mask = (y_true == class_idx)
        class_decision = y_decision[class_mask]

        axes[0].hist(class_decision, bins=20, alpha=0.6, label='True {}'.format(class_name),
                     density=True, edgecolor='black')

    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    axes[0].set_xlabel('Decision Function Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('SVM Decision Function Distribution\n(Standardized Data)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    margin_distance = np.abs(y_decision)

    axes[1].hist(margin_distance, bins=20, alpha=0.7, color='blue',
                 density=True, edgecolor='black')

    avg_margin = np.mean(margin_distance)
    axes[1].axvline(x=avg_margin, color='red', linestyle='--', linewidth=2,
                    label='Avg Margin: {:.3f}'.format(avg_margin))

    axes[1].set_xlabel('Distance to Decision Boundary', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('SVM Margin Distribution (Standardized)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("SVM decision boundary analysis saved to: {}".format(save_path))

    plt.show()


def save_fold_results_to_csv(all_results):
    print("\n" + "=" * 60)
    print("Saving Cross-Validation Fold Results to CSV")
    print("=" * 60)

    fold_summary_rows = []

    for result in all_results:
        class_pair = result['class_pair']
        class1_name = result['class1_name']
        class2_name = result['class2_name']

        for fold_info in result['cv_fold_details']:
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
    all_folds_df.to_csv('SVM_all_folds_detailed.csv', index=False)
    print("  Saved all folds detailed results to: SVM_all_folds_detailed.csv")

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
            'Support_Vectors': result['n_support_vectors'],
            'SVM_Kernel': result['svm_parameters']['kernel'],
            'SVM_C': result['svm_parameters']['C'],
            'SVM_Gamma': result['svm_parameters']['gamma']
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('SVM_overall_summary.csv', index=False)
    print("  Saved overall summary to: SVM_overall_summary.csv")

    for result in all_results:
        class_pair = result['class_pair']
        fold_rows = []
        for fold_info in result['cv_fold_details']:
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
        filename = 'SVM_fold_results_{}.csv'.format(class_pair)
        fold_df.to_csv(filename, index=False)
        print("  Saved fold results for {} to: {}".format(class_pair, filename))

    print("\nAll cross-validation fold results saved to CSV files")


def main():
    print("=" * 80)
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION ANALYSIS")
    print("=" * 80)
    print("Note: All data is standardized using Z-score before classification")
    print("=" * 80)

    print("\nLoading and standardizing data...")
    X, y, subject_ids, class_labels = load_and_prepare_data()

    if len(X) == 0:
        print("Error: No data loaded! Please check file paths.")
        return

    print("\nStarting binary classification analysis using SVM with standardized data...")

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

    print("Will perform the following 4 classification analyses:")
    for i, (class1, class2) in enumerate(class_pairs):
        print("  {}. {} vs {}".format(i + 1, class_labels[class1], class_labels[class2]))

    all_results = []

    for (class1, class2), pair_name in zip(class_pairs, pair_names):
        print("\n" + "=" * 80)
        print("PROCESSING: {} vs {}".format(class_labels[class1], class_labels[class2]))
        print("=" * 80)

        results = evaluate_two_class_classification(
            class1, class2, X, y, subject_ids, class_labels
        )
        all_results.append(results)

        svm_analysis_path = "svm_analysis_{}_vs_{}.png".format(class_labels[class1], class_labels[class2])
        plot_svm_decision_boundary_analysis(results,
                                            [class_labels[class1], class_labels[class2]],
                                            svm_analysis_path)

    save_fold_results_to_csv(all_results)

    plot_comprehensive_results(all_results, model_name="SVM")

    print("\n" + "=" * 100)
    print("OVERALL RESULTS SUMMARY - SUPPORT VECTOR MACHINE (Standardized Data)")
    print("=" * 100)
    print("{:<25} {:<15} {:<15} {:<15} {:<25}".format(
        'Classification Task', 'Train Acc', 'Test Acc', 'CV Acc (±std)', 'CV F1 (±std)'))
    print("-" * 100)

    for result in all_results:
        print("{:<25} {:<15.4f} {:<15.4f} {:.4f} (±{:.4f})  {:.4f} (±{:.4f})".format(
            result['class_pair'],
            result['train_accuracy'],
            result['test_accuracy'],
            result['cv_mean'],
            result['cv_std'],
            result['cv_f1_mean'],
            result['cv_f1_std']))

    detailed_results = []
    for result in all_results:
        cm = result['confusion_matrix']
        report = result['class_report']

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        svm_params = result.get('svm_parameters', {})
        n_support_vectors = result.get('n_support_vectors', 0)

        detailed_result = {
            'class_pair': result['class_pair'],
            'train_accuracy': result['train_accuracy'],
            'train_f1': result['train_f1'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'cv_f1_mean': result['cv_f1_mean'],
            'cv_f1_std': result['cv_f1_std'],
            'cv_scores': str([round(s, 4) for s in result['cv_scores']]),
            'cv_f1_scores': str([round(s, 4) for s in result['cv_f1_scores']]),
            'svm_kernel': svm_params.get('kernel', ''),
            'svm_C': svm_params.get('C', 0),
            'svm_gamma': svm_params.get('gamma', ''),
            'n_support_vectors': n_support_vectors,
            'support_vector_ratio': n_support_vectors / len(result['y_true']) if len(result['y_true']) > 0 else 0,
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
            'data_standardized': 'Yes (Z-score)',
            'standardization_note': 'All features scaled to mean=0, std=1 before classification'
        }
        detailed_results.append(detailed_result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('SVM_detailed_results.csv', index=False)
    print("\nDetailed SVM statistical results saved to: SVM_detailed_results.csv")

    simple_df = detailed_df[['class_pair', 'train_accuracy', 'train_f1', 'test_accuracy', 'test_f1',
                             'cv_mean', 'cv_std', 'cv_f1_mean', 'cv_f1_std',
                             'svm_kernel', 'svm_C', 'n_support_vectors', 'data_standardized']]
    simple_df.to_csv('SVM_summary_results.csv', index=False)
    print("Concise SVM results saved to: SVM_summary_results.csv")

    analyze_svm_performance(all_results)

    return all_results, detailed_df


def plot_comprehensive_results(all_results, model_name="SVM"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    class_pairs = [result['class_pair'] for result in all_results]
    test_accuracies = [result['test_accuracy'] for result in all_results]
    train_accuracies = [result['train_accuracy'] for result in all_results]
    cv_means = [result['cv_mean'] for result in all_results]
    cv_stds = [result['cv_std'] for result in all_results]

    train_f1_scores = [result['train_f1'] for result in all_results]
    test_f1_scores = [result['test_f1'] for result in all_results]
    cv_f1_means = [result['cv_f1_mean'] for result in all_results]
    cv_f1_stds = [result['cv_f1_std'] for result in all_results]

    n_support_vectors = []
    for result in all_results:
        n_sv = result.get('n_support_vectors', 0)
        n_support_vectors.append(n_sv)

    x = np.arange(len(class_pairs))
    width = 0.25

    axes[0].bar(x - width, train_accuracies, width, label='Training Accuracy', color='skyblue', alpha=0.8)
    axes[0].bar(x, test_accuracies, width, label='Test Accuracy', color='lightgreen', alpha=0.8)
    axes[0].bar(x + width, cv_means, width, label='CV Accuracy', color='salmon', alpha=0.8, yerr=cv_stds,
                capsize=5)

    axes[0].set_xlabel('Classification Task', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('{} Accuracy Comparison (Standardized Data)'.format(model_name), fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_pairs, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])

    for i, (train_acc, test_acc, cv_acc) in enumerate(zip(train_accuracies, test_accuracies, cv_means)):
        axes[0].text(i - width, train_acc + 0.02, '{:.3f}'.format(train_acc), ha='center', va='bottom', fontsize=9)
        axes[0].text(i, test_acc + 0.02, '{:.3f}'.format(test_acc), ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width, cv_acc + 0.02, '{:.3f}'.format(cv_acc), ha='center', va='bottom', fontsize=9)

    axes[1].bar(x - width, train_f1_scores, width, label='Training F1', color='#90CAF9', alpha=0.8)
    axes[1].bar(x, test_f1_scores, width, label='Test F1', color='#81C784', alpha=0.8)
    axes[1].bar(x + width, cv_f1_means, width, label='CV F1', color='#FFAB91', alpha=0.8, yerr=cv_f1_stds, capsize=5)

    axes[1].set_xlabel('Classification Task', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('{} F1 Score Comparison (Standardized Data)'.format(model_name), fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_pairs, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    for i, (train_f1, test_f1, cv_f1) in enumerate(zip(train_f1_scores, test_f1_scores, cv_f1_means)):
        axes[1].text(i - width, train_f1 + 0.02, '{:.3f}'.format(train_f1), ha='center', va='bottom', fontsize=9)
        axes[1].text(i, test_f1 + 0.02, '{:.3f}'.format(test_f1), ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width, cv_f1 + 0.02, '{:.3f}'.format(cv_f1), ha='center', va='bottom', fontsize=9)

    if n_support_vectors and any(n_sv > 0 for n_sv in n_support_vectors):
        support_ratios = []
        for result in all_results:
            n_sv = result.get('n_support_vectors', 0)
            n_samples = len(result['y_true'])
            ratio = n_sv / n_samples if n_samples > 0 else 0
            support_ratios.append(ratio)

        x_pos = np.arange(len(class_pairs))
        width = 0.35

        axes[2].bar(x_pos, n_support_vectors, width, label='Number of Support Vectors', color='skyblue', alpha=0.8)
        axes[2].set_xlabel('Classification Task', fontsize=12)
        axes[2].set_ylabel('Number of Support Vectors', fontsize=12)
        axes[2].set_title('{} Support Vectors Analysis'.format(model_name), fontsize=14, fontweight='bold')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(class_pairs, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)

        for i, n_sv in enumerate(n_support_vectors):
            axes[2].text(i, n_sv + max(n_support_vectors) * 0.01, '{}'.format(n_sv),
                         ha='center', va='bottom', fontsize=9)

        ax2 = axes[2].twinx()
        ax2.plot(x_pos, support_ratios, 'b-o', linewidth=2, markersize=8, label='Support Vector Ratio')
        ax2.set_ylabel('Support Vector Ratio', fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim([0, 1.1])

        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        axes[2].errorbar(class_pairs, cv_means, yerr=cv_stds, fmt='o-',
                         color='blue', linewidth=2, markersize=8, capsize=5, capthick=2)
        axes[2].set_xlabel('Classification Task', fontsize=12)
        axes[2].set_ylabel('Cross-validation Accuracy', fontsize=12)
        axes[2].set_title('Cross-validation Results (Standardized Data)', fontsize=14, fontweight='bold')
        axes[2].set_xticklabels(class_pairs, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.1])

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

            summary_data.append([recall_class0, precision_class0])

    if summary_data:
        summary_data = np.array(summary_data)
        x_pos = np.arange(len(class_pairs))
        width = 0.35

        axes[3].bar(x_pos - width / 2, summary_data[:, 0], width, label='Class 0 Recall', color='lightblue')
        axes[3].bar(x_pos + width / 2, summary_data[:, 1], width, label='Class 0 Precision', color='steelblue')

        axes[3].set_xlabel('Classification Task', fontsize=12)
        axes[3].set_ylabel('Score', fontsize=12)
        axes[3].set_title('{} Class 0 Performance Metrics'.format(model_name), fontsize=14, fontweight='bold')
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(class_pairs, rotation=45, ha='right')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1.1])

        for i, (recall, precision) in enumerate(summary_data):
            axes[3].text(i - width / 2, recall + 0.02, '{:.3f}'.format(recall), ha='center', va='bottom', fontsize=9)
            axes[3].text(i + width / 2, precision + 0.02, '{:.3f}'.format(precision), ha='center', va='bottom',
                         fontsize=9)

    plt.tight_layout()
    plt.savefig('svm_comprehensive_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive {} results plot saved to: svm_comprehensive_results.png".format(model_name))
    plt.show()


def analyze_svm_performance(all_results):
    print("\n" + "=" * 80)
    print("SVM CLASSIFIER SPECIFIC ANALYSIS (Standardized Data)")
    print("=" * 80)

    for result in all_results:
        y_decision = result['y_decision']
        y_true = result['y_true']
        n_sv = result.get('n_support_vectors', 0)

        margin_distance = np.abs(y_decision)

        print("\n{}:".format(result['class_pair']))
        print("  Number of support vectors: {}".format(n_sv))
        print("  Support vector ratio: {:.2%}".format(n_sv / len(y_true) if len(y_true) > 0 else 0))
        print("  Average margin distance: {:.4f}".format(np.mean(margin_distance)))
        print("  Margin standard deviation: {:.4f}".format(np.std(margin_distance)))
        print("  CV F1 Score: {:.4f} (±{:.4f})".format(result['cv_f1_mean'], result['cv_f1_std']))
        print("  CV Scores (5 folds): {}".format([round(s, 4) for s in result['cv_scores']]))
        print("  CV F1 Scores (5 folds): {}".format([round(s, 4) for s in result['cv_f1_scores']]))

        correct_mask = (y_true == result['y_pred'])
        correct_margin = margin_distance[correct_mask]
        incorrect_margin = margin_distance[~correct_mask]

        if len(correct_margin) > 0:
            print("  Correct predictions - Avg margin: {:.4f}, Min margin: {:.4f}".format(
                np.mean(correct_margin), np.min(correct_margin)))

        if len(incorrect_margin) > 0:
            print("  Incorrect predictions - Avg margin: {:.4f}, Max margin: {:.4f}".format(
                np.mean(incorrect_margin), np.max(incorrect_margin)))


if __name__ == "__main__":
    print("=" * 80)
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION ANALYSIS")
    print("4 Specified Binary Classification Tasks")
    print("All data is standardized using Z-score before analysis")
    print("=" * 80)

    all_results, detailed_df = main()