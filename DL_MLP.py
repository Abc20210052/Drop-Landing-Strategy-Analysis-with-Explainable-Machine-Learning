import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import os
import time

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
    print("✓ Neural networks perform best with standardized data")

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

def evaluate_two_class_classification_ann(class1, class2, X, y, subject_ids, class_labels,
                                          hidden_layer_sizes=(100, 50), max_iter=500):
    print("\n" + "=" * 60)
    print("Neural Network Classification: {} vs {}".format(class_labels[class1], class_labels[class2]))
    print("MLP with hidden layers: {}, max_iter={}".format(hidden_layer_sizes, max_iter))
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
    print("✓ Neural networks require standardized data for optimal performance")

    X_train_scaled = X_train
    X_test_scaled = X_test

    print("\nTraining Neural Network (MLP)...")
    start_time = time.time()

    ann_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )

    ann_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final iteration: {ann_model.n_iter_} (max: {max_iter})")

    y_train_pred = ann_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='binary')

    y_test_pred = ann_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='binary')

    y_test_proba = ann_model.predict_proba(X_test_scaled)

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
    cv_fold_details = []
    cv_training_losses = []
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

        ann_cv = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            early_stopping=True,
            random_state=42,
            verbose=False
        )
        ann_cv.fit(X_cv_train, y_cv_train)

        y_cv_pred = ann_cv.predict(X_cv_val)
        cv_score = accuracy_score(y_cv_val, y_cv_pred)
        cv_f1 = f1_score(y_cv_val, y_cv_pred, average='binary')

        cv_scores.append(cv_score)
        cv_f1_scores.append(cv_f1)
        cv_training_losses.append(ann_cv.loss_)

        cm_fold = confusion_matrix(y_cv_val, y_cv_pred)

        fold_info = {
            'fold': fold,
            'train_subjects': len(train_subjects),
            'val_subjects': len(val_subjects),
            'train_samples': np.sum(train_mask),
            'val_samples': np.sum(val_mask),
            'accuracy': cv_score,
            'f1_score': cv_f1,
            'final_loss': ann_cv.loss_,
            'n_iterations': ann_cv.n_iter_,
            'cm_00': cm_fold[0, 0] if cm_fold.shape[0] > 0 else 0,
            'cm_01': cm_fold[0, 1] if cm_fold.shape[1] > 1 else 0,
            'cm_10': cm_fold[1, 0] if cm_fold.shape[0] > 1 else 0,
            'cm_11': cm_fold[1, 1] if cm_fold.shape[0] > 1 and cm_fold.shape[1] > 1 else 0
        }
        cv_fold_details.append(fold_info)

        print("  Fold {}: Train subjects={}, Val subjects={}, Train samples={}, Val samples={}".format(
            fold, len(train_subjects), len(val_subjects),
            np.sum(train_mask), np.sum(val_mask)))
        print("    Accuracy = {:.4f}, F1 Score = {:.4f}, Final Loss = {:.4f}".format(cv_score, cv_f1, ann_cv.loss_))
        print("    Confusion Matrix:")
        print("      [[{}  {}]".format(cm_fold[0, 0], cm_fold[0, 1] if cm_fold.shape[1] > 1 else 0))
        print("       [{}  {}]]".format(cm_fold[1, 0] if cm_fold.shape[0] > 1 else 0,
                                        cm_fold[1, 1] if cm_fold.shape[0] > 1 and cm_fold.shape[1] > 1 else 0))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_f1_mean = np.mean(cv_f1_scores)
    cv_f1_std = np.std(cv_f1_scores)
    cv_loss_mean = np.mean(cv_training_losses)

    print("-" * 60)
    print("Cross-Validation Summary:")
    print("  Mean Accuracy = {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("  Mean F1 Score = {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))
    print("  Mean Final Loss = {:.4f}".format(cv_loss_mean))
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
    print("ROC AUC Score: {:.4f}".format(roc_auc))
    print("Cross-validation Mean Accuracy: {:.4f} (±{:.4f})".format(cv_mean, cv_std))
    print("Cross-validation Mean F1 Score: {:.4f} (±{:.4f})".format(cv_f1_mean, cv_f1_std))
    print("Cross-validation Mean Loss: {:.4f}".format(cv_loss_mean))

    print("\nNeural Network Model Information:")
    print("  Hidden layer sizes: {}".format(ann_model.hidden_layer_sizes))
    print("  Activation function: {}".format(ann_model.activation))
    print("  Solver: {}".format(ann_model.solver))
    print("  Alpha (L2 regularization): {}".format(ann_model.alpha))
    print("  Learning rate: {}".format(ann_model.learning_rate))
    print("  Number of iterations: {}".format(ann_model.n_iter_))
    print("  Number of layers: {}".format(ann_model.n_layers_))
    print("  Number of outputs: {}".format(ann_model.n_outputs_))
    print("  Number of features: {}".format(ann_model.n_features_in_))
    print("  Final loss: {:.6f}".format(ann_model.loss_))

    print("\nNetwork Architecture:")
    layer_units = [ann_model.n_features_in_] + list(ann_model.hidden_layer_sizes) + [ann_model.n_outputs_]
    for i, (n_in, n_out) in enumerate(zip(layer_units[:-1], layer_units[1:])):
        print(f"  Layer {i}: {n_in} inputs → {n_out} neurons")

    total_params = 0
    for i in range(len(layer_units) - 1):
        params = layer_units[i] * layer_units[i + 1] + layer_units[i + 1]
        total_params += params
    print(f"  Total parameters: {total_params}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=[class_labels[class1], class_labels[class2]]))

    print("Confusion Matrix (Counts):")
    print(cm)

    plot_title = "Neural Network Confusion Matrix: {} vs {}\n(Z-score Standardized, 4:1 Split)".format(
        class_labels[class1], class_labels[class2])
    save_path = "ann_confusion_matrix_{}_vs_{}_layers{}.png".format(
        class_labels[class1], class_labels[class2], hidden_layer_sizes)

    plot_confusion_matrix_ann(cm,
                              [class_labels[class1], class_labels[class2]],
                              plot_title,
                              save_path)

    if hasattr(ann_model, 'loss_curve_'):
        print("\nTraining History Analysis:")
        print("  Initial loss: {:.4f}".format(ann_model.loss_curve_[0]))
        print("  Final loss: {:.4f}".format(ann_model.loss_curve_[-1]))
        print("  Loss reduction: {:.2%}".format(
            (ann_model.loss_curve_[0] - ann_model.loss_curve_[-1]) / ann_model.loss_curve_[0]))

        if hasattr(ann_model, 'validation_scores_'):
            best_val_score = np.max(ann_model.validation_scores_)
            print("  Best validation score: {:.4f}".format(best_val_score))

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
        'cv_fold_details': cv_fold_details,
        'confusion_matrix': cm,
        'class_report': classification_report(y_test, y_test_pred,
                                              target_names=[class_labels[class1], class_labels[class2]],
                                              output_dict=True),
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'model': ann_model,
        'hidden_layer_sizes': hidden_layer_sizes,
        'max_iter': max_iter,
        'training_time': training_time,
        'n_iterations': ann_model.n_iter_,
        'final_loss': ann_model.loss_,
        'total_params': total_params,
        'layer_units': layer_units,
        'loss_curve': ann_model.loss_curve_ if hasattr(ann_model, 'loss_curve_') else None,
        'validation_scores': ann_model.validation_scores_ if hasattr(ann_model, 'validation_scores_') else None,
        'standardization_method': 'Z-score (StandardScaler)',
        'cv_loss_mean': cv_loss_mean,
        'data_split': '4:1 Subject-based',
        'split_ratio': '80% Training, 20% Testing'
    }

    return results

def plot_confusion_matrix_ann(cm, class_names, title, save_path=None):
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

    sns.heatmap(cm_percent, annot=annot_matrix, fmt='', cmap='Purples',
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to: {}".format(save_path))

    plt.show()

def save_fold_results_to_csv_ann(all_results, hidden_layer_sizes):
    print("\n" + "=" * 60)
    print("Saving Cross-Validation Fold Results to CSV (ANN)")
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
                'Final_Loss': fold_info['final_loss'],
                'Iterations': fold_info['n_iterations'],
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_summary_rows.append(row)

    all_folds_df = pd.DataFrame(fold_summary_rows)
    all_folds_df.to_csv('ANN_all_folds_detailed_layers{}.csv'.format(hidden_layer_sizes), index=False)
    print("  Saved all folds detailed results to: ANN_all_folds_detailed_layers{}.csv".format(hidden_layer_sizes))

    summary_rows = []
    for result in all_results:
        row = {
            'Classification_Task': result['class_pair'],
            'Class1': result['class1_name'],
            'Class2': result['class2_name'],
            'Hidden_Layers': str(result['hidden_layer_sizes']),
            'Train_Accuracy': result['train_accuracy'],
            'Train_F1': result['train_f1'],
            'Test_Accuracy': result['test_accuracy'],
            'Test_F1': result['test_f1'],
            'ROC_AUC': result['roc_auc'],
            'CV_Mean_Accuracy': result['cv_mean'],
            'CV_Std_Accuracy': result['cv_std'],
            'CV_Mean_F1': result['cv_f1_mean'],
            'CV_Std_F1': result['cv_f1_std'],
            'CV_Mean_Loss': result['cv_loss_mean'],
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
            'Final_Loss': result['final_loss'],
            'Total_Parameters': result['total_params'],
            'N_Layers': len(result['layer_units']),
            'N_Iterations': result['n_iterations'],
            'Standardization': result['standardization_method'],
            'Data_Split': result['data_split']
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('ANN_overall_summary_layers{}.csv'.format(hidden_layer_sizes), index=False)
    print("  Saved overall summary to: ANN_overall_summary_layers{}.csv".format(hidden_layer_sizes))

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
                'Final_Loss': fold_info['final_loss'],
                'Iterations': fold_info['n_iterations'],
                'Confusion_Matrix_00': fold_info['cm_00'],
                'Confusion_Matrix_01': fold_info['cm_01'],
                'Confusion_Matrix_10': fold_info['cm_10'],
                'Confusion_Matrix_11': fold_info['cm_11']
            }
            fold_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        filename = 'ANN_fold_results_{}_layers{}.csv'.format(class_pair, hidden_layer_sizes)
        fold_df.to_csv(filename, index=False)
        print("  Saved fold results for {} to: {}".format(class_pair, filename))

    print("\n✓ All cross-validation fold results saved to CSV files")

def plot_ann_training_history(results, save_path=None):
    model = results['model']
    class_pair = results['class_pair']

    if hasattr(model, 'loss_curve_'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(model.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Neural Network Training Loss\n{class_pair}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        final_loss = model.loss_curve_[-1]
        axes[0].text(0.98, 0.98, f'Final Loss: {final_loss:.4f}',
                     transform=axes[0].transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if hasattr(model, 'validation_scores_'):
            axes[1].plot(model.validation_scores_, 'r-', linewidth=2, label='Validation Score')
            axes[1].set_xlabel('Iteration', fontsize=12)
            axes[1].set_ylabel('Validation Score', fontsize=12)
            axes[1].set_title(f'Validation Score During Training\n{class_pair}', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            best_val_score = np.max(model.validation_scores_)
            best_iter = np.argmax(model.validation_scores_)
            axes[1].text(0.98, 0.98, f'Best Score: {best_val_score:.4f}\nat iteration {best_iter}',
                         transform=axes[1].transAxes, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1].remove()
            ax2 = fig.add_subplot(122)

            loss_reduction = (model.loss_curve_[0] - model.loss_curve_[-1]) / model.loss_curve_[0]
            ax2.bar(['Initial', 'Final'], [model.loss_curve_[0], model.loss_curve_[-1]],
                    color=['red', 'green'], alpha=0.7)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Loss Reduction: {loss_reduction:.1%}\n{class_pair}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            for i, (label, value) in enumerate(
                    zip(['Initial', 'Final'], [model.loss_curve_[0], model.loss_curve_[-1]])):
                ax2.text(i, value + max(model.loss_curve_) * 0.01, f'{value:.4f}',
                         ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Neural Network training history plot saved to: {save_path}")

        plt.show()
    else:
        print("No training history available for plotting.")

def plot_ann_probability_analysis(results, class_labels_pair, save_path=None):
    y_true = results['y_true']
    y_proba = results['y_proba']
    y_pred = results['y_pred']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for class_idx, class_name in enumerate(class_labels_pair):
        class_mask = (y_true == class_idx)
        class_proba = y_proba[class_mask, class_idx]

        axes[0, 0].hist(class_proba, bins=20, alpha=0.6, label=f'True {class_name}',
                        density=True, edgecolor='black')

    axes[0, 0].set_xlabel('Predicted Probability for True Class', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Probability Distribution by True Class\n(Higher = more confident)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    epsilon = 1e-10
    entropy = -np.sum(y_proba * np.log(y_proba + epsilon), axis=1)

    axes[0, 1].hist(entropy, bins=20, alpha=0.7, color='purple',
                    density=True, edgecolor='black')

    avg_entropy = np.mean(entropy)
    axes[0, 1].axvline(x=avg_entropy, color='red', linestyle='--', linewidth=2,
                       label=f'Avg Entropy: {avg_entropy:.3f}')

    axes[0, 1].set_xlabel('Prediction Entropy', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Prediction Uncertainty (Entropy)\n(Higher = more uncertain)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    max_proba = np.max(y_proba, axis=1)
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
    axes[1, 0].plot(bin_centers, accuracy_by_bin, 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].fill_between(bin_centers, accuracy_by_bin, alpha=0.3, color='purple')

    for i, (center, acc, n_samples) in enumerate(zip(bin_centers, accuracy_by_bin, samples_by_bin)):
        if n_samples > 0:
            axes[1, 0].text(center, acc + 0.02, f'n={n_samples}',
                            ha='center', va='bottom', fontsize=8)

    axes[1, 0].set_xlabel('Maximum Probability (Confidence)', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Accuracy vs. Prediction Confidence\n(Calibration Analysis)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    correct_proba = max_proba[correct_mask]
    incorrect_proba = max_proba[~correct_mask]

    data_to_plot = [correct_proba, incorrect_proba] if len(incorrect_proba) > 0 else [correct_proba]
    labels = ['Correct', 'Incorrect'] if len(incorrect_proba) > 0 else ['Correct']

    bp = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)

    colors = ['lightgreen', 'lightcoral'] if len(incorrect_proba) > 0 else ['lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1, 1].set_ylabel('Maximum Probability', fontsize=12)
    axes[1, 1].set_title('Prediction Confidence:\nCorrect vs Incorrect Predictions',
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Neural Network probability analysis plot saved to: {save_path}")

    plt.show()

def plot_ann_architecture_comparison(X, y, subject_ids, class_labels, class_pairs,
                                     architectures=[(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]):
    print("\n" + "=" * 60)
    print("Neural Network Architecture Comparison (4:1 Split)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    results_by_arch = {arch: [] for arch in architectures}

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

        for arch in architectures:
            print(f"  Architecture {arch}: ", end='')

            start_time = time.time()
            ann = MLPClassifier(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=300,
                early_stopping=True,
                random_state=42,
                verbose=False
            )
            ann.fit(X_train, y_train)

            training_time = time.time() - start_time

            y_pred = ann.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='binary')

            test_accuracies.append(test_acc)
            test_f1_scores.append(test_f1)
            training_times.append(training_time)

            results_by_arch[arch].append(test_acc)

            print(f"Test Acc={test_acc:.4f}, F1={test_f1:.4f}, Time={training_time:.2f}s")

        x_pos = np.arange(len(architectures))
        width = 0.35

        axes[idx].bar(x_pos - width / 2, test_accuracies, width,
                      label='Test Accuracy', color='purple', alpha=0.8)
        axes[idx].bar(x_pos + width / 2, test_f1_scores, width,
                      label='Test F1 Score', color='mediumpurple', alpha=0.8)

        axes[idx].set_xlabel('Network Architecture', fontsize=12)
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].set_title(f'{class_labels[class1]} vs {class_labels[class2]}\n(4:1 Split)',
                            fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels([str(arch) for arch in architectures], rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1.1])

        for i, (acc, f1) in enumerate(zip(test_accuracies, test_f1_scores)):
            axes[idx].text(i - width / 2, acc + 0.02, f'{acc:.3f}',
                           ha='center', va='bottom', fontsize=8)
            axes[idx].text(i + width / 2, f1 + 0.02, f'{f1:.3f}',
                           ha='center', va='bottom', fontsize=8)

        best_idx = np.argmax(test_f1_scores)
        best_arch = architectures[best_idx]
        best_f1 = test_f1_scores[best_idx]

        axes[idx].text(0.02, 0.95, f'Best (F1): {best_arch}\nF1: {best_f1:.4f}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('ann_architecture_comparison_4to1.png', dpi=300, bbox_inches='tight')
    print("\nNeural Network architecture comparison plot saved to: ann_architecture_comparison_4to1.png")
    plt.show()

    print("\n" + "=" * 60)
    print("ARCHITECTURE PERFORMANCE SUMMARY (4:1 Split)")
    print("=" * 60)
    print("{:<20} {:<10} {:<10} {:<10}".format(
        'Architecture', 'Mean Acc', 'Std Acc', 'Mean F1'))
    print("-" * 60)

    for arch in architectures:
        accuracies = results_by_arch[arch]
        mean_acc = np.mean(accuracies) if accuracies else 0
        std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            str(arch), mean_acc, std_acc, mean_acc))

def main_ann_analysis():
    print("=" * 80)
    print("ARTIFICIAL NEURAL NETWORK (ANN) CLASSIFICATION ANALYSIS")
    print("Multilayer Perceptron (MLP)")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("=" * 80)

    print("\nLoading and standardizing data...")
    X, y, subject_ids, repetition_ids, class_labels = load_and_prepare_data()

    if len(X) == 0:
        print("Error: No data loaded! Please check file paths.")
        return

    hidden_layer_sizes = (100, 50)
    max_iter = 500

    print(f"\nUsing Neural Network with hidden layers: {hidden_layer_sizes}, max_iter={max_iter}")
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

        results = evaluate_two_class_classification_ann(
            class1, class2, X, y, subject_ids, class_labels,
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter
        )

        all_results.append(results)

        training_history_path = "ann_training_history_{}_vs_{}_4to1.png".format(
            class_labels[class1], class_labels[class2])
        plot_ann_training_history(results, save_path=training_history_path)

        proba_analysis_path = "ann_probability_analysis_{}_vs_{}_4to1.png".format(
            class_labels[class1], class_labels[class2])
        plot_ann_probability_analysis(results,
                                      [class_labels[class1], class_labels[class2]],
                                      proba_analysis_path)

    save_fold_results_to_csv_ann(all_results, hidden_layer_sizes)

    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT NEURAL NETWORK ARCHITECTURES (4:1 Split)")
    print("=" * 80)
    plot_ann_architecture_comparison(X, y, subject_ids, class_labels, class_pairs)

    plot_comprehensive_results_ann(all_results, hidden_layer_sizes)

    print("\n" + "=" * 100)
    print("OVERALL RESULTS SUMMARY - NEURAL NETWORK (MLP)")
    print(f"Architecture: {hidden_layer_sizes}")
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 100)
    print("{:<25} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
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

    detailed_results = []
    for result in all_results:
        cm = result['confusion_matrix']
        report = result['class_report']

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        detailed_result = {
            'class_pair': result['class_pair'],
            'hidden_layers': str(result['hidden_layer_sizes']),
            'max_iter': max_iter,
            'actual_iterations': result['n_iterations'],
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
            'training_time': result['training_time'],
            'final_loss': result['final_loss'],
            'cv_loss_mean': result.get('cv_loss_mean', 0),
            'total_parameters': result['total_params'],
            'n_layers': len(result['layer_units']),
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
            'split_ratio': '80% Training, 20% Testing'
        }
        detailed_results.append(detailed_result)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('ANN_detailed_results_4to1_layers{}.csv'.format(hidden_layer_sizes), index=False)
    print("\nDetailed Neural Network statistical results saved to: ANN_detailed_results_4to1_layers{}.csv".format(
        hidden_layer_sizes))

    simple_df = detailed_df[['class_pair', 'hidden_layers', 'train_accuracy', 'train_f1',
                             'test_accuracy', 'test_f1', 'roc_auc',
                             'cv_mean', 'cv_std', 'cv_f1_mean', 'cv_f1_std',
                             'training_time', 'final_loss', 'total_parameters',
                             'standardization_method', 'data_split']]
    simple_df.to_csv('ANN_summary_results_4to1_layers{}.csv'.format(hidden_layer_sizes), index=False)
    print("Concise Neural Network results saved to: ANN_summary_results_4to1_layers{}.csv".format(hidden_layer_sizes))

    analyze_ann_performance(all_results, hidden_layer_sizes)

    return all_results, detailed_df

def plot_comprehensive_results_ann(all_results, hidden_layer_sizes):
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
    final_losses = [result['final_loss'] for result in all_results]

    x = np.arange(len(class_pairs))
    width = 0.25

    axes[0].bar(x - width, train_accuracies, width, label='Training Accuracy', color='lavender', alpha=0.8)
    axes[0].bar(x, test_accuracies, width, label='Test Accuracy', color='mediumpurple', alpha=0.8)
    axes[0].bar(x + width, cv_means, width, label='CV Accuracy', color='rebeccapurple', alpha=0.8,
                yerr=cv_stds, capsize=5)

    axes[0].set_xlabel('Classification Task', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Neural Network Accuracy Comparison\n(4:1 Split, Z-score Standardized)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_pairs, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])

    for i, (train_acc, test_acc, cv_acc) in enumerate(zip(train_accuracies, test_accuracies, cv_means)):
        axes[0].text(i - width, train_acc + 0.02, '{:.3f}'.format(train_acc),
                     ha='center', va='bottom', fontsize=9)
        axes[0].text(i, test_acc + 0.02, '{:.3f}'.format(test_acc),
                     ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width, cv_acc + 0.02, '{:.3f}'.format(cv_acc),
                     ha='center', va='bottom', fontsize=9)

    axes[1].bar(x - width, train_f1_scores, width, label='Training F1', color='#D8BFD8', alpha=0.8)
    axes[1].bar(x, test_f1_scores, width, label='Test F1', color='#9370DB', alpha=0.8)
    axes[1].bar(x + width, cv_f1_means, width, label='CV F1', color='#6A0DAD', alpha=0.8,
                yerr=cv_f1_stds, capsize=5)

    axes[1].set_xlabel('Classification Task', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Neural Network F1 Score Comparison\n(4:1 Split, Z-score Standardized)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_pairs, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    for i, (train_f1, test_f1, cv_f1) in enumerate(zip(train_f1_scores, test_f1_scores, cv_f1_means)):
        axes[1].text(i - width, train_f1 + 0.02, '{:.3f}'.format(train_f1),
                     ha='center', va='bottom', fontsize=9)
        axes[1].text(i, test_f1 + 0.02, '{:.3f}'.format(test_f1),
                     ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width, cv_f1 + 0.02, '{:.3f}'.format(cv_f1),
                     ha='center', va='bottom', fontsize=9)

    x_pos = np.arange(len(class_pairs))
    width = 0.35

    time_bars = axes[2].bar(x_pos - width / 2, training_times, width,
                            label='Training Time (s)', color='lavender', alpha=0.8)

    ax2 = axes[2].twinx()
    loss_bars = ax2.bar(x_pos + width / 2, final_losses, width,
                        label='Final Loss', color='purple', alpha=0.8)

    axes[2].set_xlabel('Classification Task', fontsize=12)
    axes[2].set_ylabel('Training Time (s)', fontsize=12, color='mediumpurple')
    ax2.set_ylabel('Final Loss', fontsize=12, color='purple')
    axes[2].set_title('Training Performance Analysis\n(Time and Loss)',
                      fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(class_pairs, rotation=45, ha='right')

    axes[2].tick_params(axis='y', labelcolor='mediumpurple')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    axes[2].grid(True, alpha=0.3, axis='x')

    for i, (time_val, loss) in enumerate(zip(training_times, final_losses)):
        axes[2].text(i - width / 2, time_val + max(training_times) * 0.01, f'{time_val:.1f}',
                     ha='center', va='bottom', fontsize=9, color='mediumpurple')
        ax2.text(i + width / 2, loss + max(final_losses) * 0.01, f'{loss:.4f}',
                 ha='center', va='bottom', fontsize=9, color='purple')

    accuracy_gaps = [train - test for train, test in zip(train_accuracies, test_accuracies)]

    x_pos = np.arange(len(class_pairs))
    bars = axes[3].bar(x_pos, accuracy_gaps,
                       color=['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' for gap in accuracy_gaps],
                       alpha=0.8)

    axes[3].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate overfitting threshold')
    axes[3].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High overfitting threshold')
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    axes[3].set_xlabel('Classification Task', fontsize=12)
    axes[3].set_ylabel('Accuracy Gap (Train - Test)', fontsize=12)
    axes[3].set_title('Neural Network Overfitting Analysis\n(Large gaps indicate overfitting)',
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
                     ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig('ann_comprehensive_results_4to1_layers{}.png'.format(hidden_layer_sizes), dpi=300, bbox_inches='tight')
    print("\nComprehensive Neural Network results plot saved to: ann_comprehensive_results_4to1_layers{}.png".format(
        hidden_layer_sizes))
    plt.show()

def analyze_ann_performance(all_results, hidden_layer_sizes):
    print("\n" + "=" * 80)
    print("NEURAL NETWORK CLASSIFIER SPECIFIC ANALYSIS (4:1 Split)")
    print(f"Architecture: {hidden_layer_sizes}")
    print("Z-score Standardized, 4:1 Subject-based Split")
    print("=" * 80)

    for result in all_results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_proba = result['y_proba']
        loss_curve = result.get('loss_curve')
        n_iterations = result['n_iterations']
        training_time = result['training_time']
        total_params = result['total_params']

        correct_mask = (y_true == y_pred)

        max_proba = np.max(y_proba, axis=1)
        avg_certainty_correct = np.mean(max_proba[correct_mask]) if np.sum(correct_mask) > 0 else 0
        avg_certainty_incorrect = np.mean(max_proba[~correct_mask]) if np.sum(~correct_mask) > 0 else 0

        epsilon = 1e-10
        entropy = -np.sum(y_proba * np.log(y_proba + epsilon), axis=1)
        avg_entropy = np.mean(entropy)

        print("\n{}:".format(result['class_pair']))
        print("  Network Architecture:")
        print("    Hidden layers: {}".format(result['hidden_layer_sizes']))
        print("    Total parameters: {}".format(total_params))
        print("    Training iterations: {}".format(n_iterations))
        print("    Training time: {:.2f} seconds".format(training_time))

        if loss_curve is not None:
            loss_reduction = (loss_curve[0] - loss_curve[-1]) / loss_curve[0]
            print("    Loss reduction: {:.1%} ({} → {})".format(
                loss_reduction, loss_curve[0], loss_curve[-1]))

        print("\n  Prediction Analysis:")
        print("    Correct predictions: {}/{} ({:.1%})".format(
            np.sum(correct_mask), len(y_true), np.sum(correct_mask) / len(y_true)))
        print("    Average prediction entropy: {:.4f}".format(avg_entropy))
        print("    F1 Score:")
        print("      Training F1: {:.4f}".format(result['train_f1']))
        print("      Test F1: {:.4f}".format(result['test_f1']))
        print("      CV F1: {:.4f} (±{:.4f})".format(result['cv_f1_mean'], result['cv_f1_std']))
        print("    Certainty analysis:")
        print("      Correct predictions - Avg certainty: {:.4f}".format(avg_certainty_correct))
        print("      Incorrect predictions - Avg certainty: {:.4f}".format(avg_certainty_incorrect))

        if avg_certainty_incorrect > 0 and avg_certainty_correct > 0:
            certainty_ratio = avg_certainty_incorrect / avg_certainty_correct
            if certainty_ratio > 1.0:
                print("      ⚠️  Warning: Incorrect predictions have higher certainty than correct ones")
                print("        This indicates model overconfidence - consider regularization")

        train_acc = result['train_accuracy']
        test_acc = result['test_accuracy']
        acc_gap = train_acc - test_acc

        print("\n  Overfitting Analysis:")
        print("    Training accuracy: {:.4f}".format(train_acc))
        print("    Test accuracy: {:.4f}".format(test_acc))
        print("    Accuracy gap: {:.4f}".format(acc_gap))

        if acc_gap > 0.2:
            print("    ⚠️  HIGH OVERFITTING: Large accuracy gap (>0.2)")
            print("      Suggestions: Increase L2 regularization (alpha), use dropout, or reduce network size")
        elif acc_gap > 0.15:
            print("    ⚠️  Moderate overfitting: Accuracy gap > 0.15")
            print("      Consider: Slight increase in regularization, early stopping")
        elif acc_gap > 0.1:
            print("    ⚠️  Slight overfitting: Accuracy gap > 0.1")
            print("      Acceptable for neural networks")
        else:
            print("    ✓ Good: Small accuracy gap indicates good generalization")

        print("\n  Training Efficiency:")
        acc_per_second = test_acc / training_time if training_time > 0 else 0
        print("    Accuracy per second: {:.6f}".format(acc_per_second))

        if training_time > 10:
            print("    ⚠️  Long training time: Consider reducing network size or increasing batch size")

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
        print("    CV Fold Details:")
        print("      Accuracy scores: {}".format([round(s, 4) for s in result['cv_scores']]))
        print("      F1 scores: {}".format([round(s, 4) for s in result['cv_f1_scores']]))

if __name__ == "__main__":
    print("=" * 80)
    print("ARTIFICIAL NEURAL NETWORK (ANN) CLASSIFICATION ANALYSIS")
    print("Multilayer Perceptron (MLP)")
    print("4 Binary Classification Tasks")
    print("Z-score Standardized Data")
    print("4:1 Train-Test Split (Subject-based)")
    print("Same subject data not in both train and test sets")
    print("=" * 80)

    all_results, detailed_df = main_ann_analysis()