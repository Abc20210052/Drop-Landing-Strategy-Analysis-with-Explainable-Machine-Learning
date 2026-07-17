import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import time
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

output_root = 'LSTM_CV_Analysis_Results_4to1'
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


def load_and_analyze_data(file_paths, class_labels):
    all_data = []
    all_labels = []
    all_subject_ids = []

    base_subject_ids = []
    for subject in range(20):
        base_subject_ids.extend([subject] * 3)
    base_subject_ids = np.array(base_subject_ids)

    for i, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path, header=None)
        filename = os.path.basename(file_path)
        print(f"Loaded: {filename}, Shape: {data.shape}")

        X = data.values.astype(np.float32)
        y = np.full((X.shape[0], 1), i, dtype=np.int32)
        subject_ids = base_subject_ids.copy()

        all_data.append(X)
        all_labels.append(y)
        all_subject_ids.append(subject_ids)

    X_all = np.vstack(all_data)
    y_all = np.vstack(all_labels).ravel()
    subject_ids_all = np.concatenate(all_subject_ids)

    print(f"\nTotal samples: {X_all.shape[0]}")
    print(f"Features per sample: {X_all.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    return X_all, y_all, subject_ids_all, class_labels


def create_binary_dataset(X, y, subject_ids, class_labels, pair_name):
    class_name_to_idx = {name: idx for idx, name in enumerate(class_labels)}

    if pair_name == '40-80':
        mask = (y == class_name_to_idx['DL_40']) | (y == class_name_to_idx['DL_80'])
        binary_labels = np.where(y[mask] == class_name_to_idx['DL_40'], 0, 1)
    elif pair_name == 'P40-P80':
        mask = (y == class_name_to_idx['DL_P40']) | (y == class_name_to_idx['DL_P80'])
        binary_labels = np.where(y[mask] == class_name_to_idx['DL_P40'], 0, 1)
    elif pair_name == '40-P40':
        mask = (y == class_name_to_idx['DL_40']) | (y == class_name_to_idx['DL_P40'])
        binary_labels = np.where(y[mask] == class_name_to_idx['DL_40'], 0, 1)
    elif pair_name == '80-P80':
        mask = (y == class_name_to_idx['DL_80']) | (y == class_name_to_idx['DL_P80'])
        binary_labels = np.where(y[mask] == class_name_to_idx['DL_80'], 0, 1)
    else:
        raise ValueError(f"Unknown task: {pair_name}")

    X_pair = X[mask]
    y_pair = binary_labels
    subject_ids_pair = subject_ids[mask]

    print(f"\n{pair_name} Dataset:")
    print(f"Samples: {len(X_pair)} (Class 0: {np.sum(y_pair == 0)}, Class 1: {np.sum(y_pair == 1)})")

    return X_pair, y_pair, subject_ids_pair


def create_simple_lstm_model(input_shape, n_classes=2):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        layers.LSTM(32, return_sequences=False, dropout=0.2),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


def create_cnn_lstm_model(input_shape, n_classes=2):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.LSTM(16, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


def compute_class_weights(y):
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return {i: class_weights[i] for i in range(len(class_weights))}
    except:
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {i: total / (len(unique) * count) for i, count in zip(unique, counts)}


def cross_validate_model(X, y, subject_ids, pair_name, model_type='simple_lstm', n_folds=5):
    print(f"\n{'=' * 60}")
    print(f"5-FOLD CROSS-VALIDATION - {pair_name} ({model_type.upper()})")
    print(f"{'=' * 60}")

    unique_subjects = np.unique(subject_ids)

    subject_labels = []
    for subj in unique_subjects:
        mask = (subject_ids == subj)
        subject_labels.append(y[mask][0])
    subject_labels = np.array(subject_labels)

    n_timesteps = 201
    n_features = 7
    X_2d = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_scaled_2d = scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(X.shape[0], n_timesteps, n_features)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_auc_scores = []
    fold_epochs = []
    fold_models = []
    all_y_true = []
    all_y_pred = []

    fold = 1
    for train_subj_idx, val_subj_idx in skf.split(unique_subjects, subject_labels):
        print(f"\n{'=' * 40}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'=' * 40}")

        train_subjects = unique_subjects[train_subj_idx]
        val_subjects = unique_subjects[val_subj_idx]

        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)

        X_train = X_scaled[train_mask]
        X_val = X_scaled[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]

        print(f"Training subjects: {len(train_subjects)} ({len(train_subjects) / len(unique_subjects) * 100:.0f}%)")
        print(f"Validation subjects: {len(val_subjects)} ({len(val_subjects) / len(unique_subjects) * 100:.0f}%)")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Train/Val ratio: {len(X_train)}:{len(X_val)} ≈ {len(X_train) / len(X_val):.2f}:1")

        input_shape = X_train.shape[1:]
        if model_type == 'cnn_lstm':
            model = create_cnn_lstm_model(input_shape)
        else:
            model = create_simple_lstm_model(input_shape)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
        )

        y_train_cat = keras.utils.to_categorical(y_train, 2)
        y_val_cat = keras.utils.to_categorical(y_val, 2)

        X_train_final, X_val_inner, y_train_final, y_val_inner = train_test_split(
            X_train, y_train_cat,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        class_weights = compute_class_weights(y_train)

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0,
                                    min_delta=0.001),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        ]

        print("\nTraining model...")
        start_time = time.time()

        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val_inner, y_val_inner),
            epochs=150,
            batch_size=8,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=0
        )

        training_time = time.time() - start_time
        actual_epochs = len(history.history['loss'])

        y_pred = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_val, y_pred_classes)
        f1 = f1_score(y_val, y_pred_classes, average='binary')

        if len(np.unique(y_val)) > 1:
            auc = roc_auc_score(y_val, y_pred[:, 1])
        else:
            auc = 0.0

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)
        fold_epochs.append(actual_epochs)
        fold_models.append(model)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred_classes)

        print(f"\n  Fold {fold} Results:")
        print(f"    Validation Accuracy: {acc:.4f}")
        print(f"    Validation F1 Score: {f1:.4f}")
        print(f"    Validation AUC: {auc:.4f}")
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Training epochs: {actual_epochs}")

        fold += 1

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    mean_auc = np.mean(fold_auc_scores)
    std_auc = np.std(fold_auc_scores)
    mean_epochs = np.mean(fold_epochs)

    cm = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION SUMMARY - {pair_name} ({model_type.upper()})")
    print(f"{'=' * 60}")
    print(f"\nAverage Results ({n_folds}-Fold CV):")
    print(f"  Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
    print(f"  F1 Score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"  AUC:      {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"  Average Epochs: {mean_epochs:.1f}")

    print(f"\nPer-fold Results:")
    for i, (acc, f1, auc, epochs) in enumerate(zip(fold_accuracies, fold_f1_scores, fold_auc_scores, fold_epochs), 1):
        print(f"  Fold {i}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Epochs={epochs}")

    print(f"\nOverall Confusion Matrix:")
    print(cm)

    plot_cv_results(fold_accuracies, fold_f1_scores, fold_auc_scores, cm, pair_name, model_type,
                    mean_acc, mean_f1, mean_auc, std_acc, std_f1)

    return {
        'pair_name': pair_name,
        'model_type': model_type,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'mean_epochs': mean_epochs,
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores,
        'fold_auc_scores': fold_auc_scores,
        'fold_epochs': fold_epochs,
        'confusion_matrix': cm,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'n_folds': n_folds
    }


def plot_cv_results(accuracies, f1_scores, auc_scores, cm, pair_name, model_type,
                    mean_acc, mean_f1, mean_auc, std_acc, std_f1):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    folds = np.arange(1, len(accuracies) + 1)

    axes[0, 0].bar(folds, accuracies, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_acc:.4f} (±{std_acc:.4f})')
    axes[0, 0].fill_between(folds, mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.2, color='red', label='±1 Std')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'{model_type.upper()} - CV Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    for i, (fold, acc) in enumerate(zip(folds, accuracies)):
        axes[0, 0].text(fold, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    axes[0, 1].bar(folds, f1_scores, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=mean_f1, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_f1:.4f} (±{std_f1:.4f})')
    axes[0, 1].fill_between(folds, mean_f1 - std_f1, mean_f1 + std_f1,
                            alpha=0.2, color='red', label='±1 Std')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title(f'{model_type.upper()} - CV F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])
    for i, (fold, f1) in enumerate(zip(folds, f1_scores)):
        axes[0, 1].text(fold, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

    axes[0, 2].bar(folds, auc_scores, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 2].axhline(y=mean_auc, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_auc:.4f}')
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title(f'{model_type.upper()} - CV AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.1])
    for i, (fold, auc) in enumerate(zip(folds, auc_scores)):
        axes[0, 2].text(fold, auc + 0.02, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    axes[1, 0].set_title('Overall Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    axes[1, 1].axis('off')
    info_text = f"""
    Model: {model_type.upper()}
    Task: {pair_name}
    Split: 4:1 (80% Train, 20% Test)
    CV: 5-Fold Cross-Validation

    === CV Results ===
    Accuracy:  {mean_acc:.4f} (±{std_acc:.4f})
    F1 Score:  {mean_f1:.4f} (±{std_f1:.4f})
    AUC:       {mean_auc:.4f} (±{std_auc:.4f})
    """
    axes[1, 1].text(0.02, 0.98, info_text, ha='left', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[1, 2].boxplot([accuracies, f1_scores], patch_artist=True,
                       labels=['Accuracy', 'F1 Score'])
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Performance Distribution')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim([0, 1.1])

    plt.suptitle(f'{model_type.upper()} - {pair_name} (5-Fold CV, 4:1 Split)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"lstm_cv_results_{pair_name}_{model_type}_4to1"
    save_figure(fig, filename)
    plt.show()


def save_cv_results_to_excel(all_results, output_root):
    if not all_results:
        print("No results to save")
        return

    results_list = []
    for result in all_results:
        for fold_idx, (acc, f1, auc, epochs) in enumerate(zip(
                result['fold_accuracies'],
                result['fold_f1_scores'],
                result['fold_auc_scores'],
                result['fold_epochs']
        ), 1):
            results_list.append({
                'Task': result['pair_name'],
                'Model_Type': result['model_type'],
                'Fold': fold_idx,
                'Fold_Accuracy': acc,
                'Fold_F1': f1,
                'Fold_AUC': auc,
                'Fold_Epochs': epochs,
                'CV_Mean_Accuracy': result['mean_accuracy'],
                'CV_Std_Accuracy': result['std_accuracy'],
                'CV_Mean_F1': result['mean_f1'],
                'CV_Std_F1': result['std_f1'],
                'CV_Mean_AUC': result['mean_auc'],
                'CV_Std_AUC': result['std_auc'],
                'Data_Split': '4:1 Subject-based',
                'CV_Folds': 5
            })

    df = pd.DataFrame(results_list)

    csv_path = os.path.join(output_root, 'LSTM_CV_results_4to1.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {csv_path}")

    try:
        excel_path = os.path.join(output_root, 'LSTM_CV_results_4to1.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Results', index=False)

            summary_df = df.groupby(['Task', 'Model_Type']).agg({
                'Fold_Accuracy': ['mean', 'std'],
                'Fold_F1': ['mean', 'std'],
                'Fold_AUC': ['mean', 'std']
            }).round(4)
            summary_df.to_excel(writer, sheet_name='Summary')

            best = df.loc[df.groupby('Task')['CV_Mean_Accuracy'].idxmax()]
            best[['Task', 'Model_Type', 'CV_Mean_Accuracy', 'CV_Mean_F1']].to_excel(
                writer, sheet_name='Best_Models', index=False)
        print(f"Results saved to Excel: {excel_path}")
    except:
        print("openpyxl not installed, only CSV format saved")


def plot_overall_cv_comparison(all_results):
    if not all_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tasks = sorted(set([r['pair_name'] for r in all_results]))
    models = sorted(set([r['model_type'] for r in all_results]))

    x = np.arange(len(tasks))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        means_acc = []
        stds_acc = []
        means_f1 = []
        stds_f1 = []
        for task in tasks:
            result = next((r for r in all_results if r['pair_name'] == task and r['model_type'] == model), None)
            if result:
                means_acc.append(result['mean_accuracy'])
                stds_acc.append(result['std_accuracy'])
                means_f1.append(result['mean_f1'])
                stds_f1.append(result['std_f1'])
            else:
                means_acc.append(0)
                stds_acc.append(0)
                means_f1.append(0)
                stds_f1.append(0)

        axes[0].bar(x + (i - len(models) / 2 + 0.5) * width, means_acc, width,
                    label=model, alpha=0.8, yerr=stds_acc, capsize=3)
        axes[1].bar(x + (i - len(models) / 2 + 0.5) * width, means_f1, width,
                    label=model, alpha=0.8, yerr=stds_f1, capsize=3)

    axes[0].set_xlabel('Task')
    axes[0].set_ylabel('CV Accuracy')
    axes[0].set_title('5-Fold CV Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    axes[1].set_xlabel('Task')
    axes[1].set_ylabel('CV F1 Score')
    axes[1].set_title('5-Fold CV F1 Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.1])
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('LSTM Models: 5-Fold CV Performance (4:1 Split)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = "lstm_cv_overall_comparison_4to1"
    save_figure(fig, filename)
    plt.show()


def print_final_summary(all_results):
    print("\n" + "=" * 100)
    print("FINAL SUMMARY - ALL TASKS AND MODELS (5-Fold CV, 4:1 Split)")
    print("=" * 100)

    print(f"\n{'Task':<10} {'Model':<15} {'CV Acc (Mean±Std)':<25} {'CV F1 (Mean±Std)':<25} {'CV AUC (Mean±Std)':<25}")
    print("-" * 110)

    for result in all_results:
        print(f"{result['pair_name']:<10} "
              f"{result['model_type']:<15} "
              f"{result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f} "
              f"{result['mean_f1']:.4f}±{result['std_f1']:.4f} "
              f"{result['mean_auc']:.4f}±{result['std_auc']:.4f}")

    print("\n" + "=" * 60)
    print("BEST MODELS BY TASK (based on CV Accuracy)")
    print("=" * 60)

    pair_results = {}
    for result in all_results:
        pair = result['pair_name']
        if pair not in pair_results:
            pair_results[pair] = []
        pair_results[pair].append(result)

    for pair in pair_results:
        best = max(pair_results[pair], key=lambda x: x['mean_accuracy'])
        print(f"\n{pair}:")
        print(f"  Best Model: {best['model_type']}")
        print(f"  CV Accuracy: {best['mean_accuracy']:.4f} (±{best['std_accuracy']:.4f})")
        print(f"  CV F1 Score: {best['mean_f1']:.4f} (±{best['std_f1']:.4f})")


def main():
    file_paths = [
        r'D:\DL\DATA\feature1_DL_40.csv',
        r'D:\DL\DATA\feature1_DL_80.csv',
        r'D:\DL\DATA\feature1_DL_P40.csv',
        r'D:\DL\DATA\feature1_DL_P80.csv'
    ]

    class_labels = ['DL_40', 'DL_80', 'DL_P40', 'DL_P80']

    print("=" * 60)
    print("LSTM CLASSIFICATION WITH 5-FOLD CV")
    print("4 Binary Classification Tasks")
    print("4:1 Subject-based Split")
    print("5-Fold Cross-Validation")
    print("=" * 60)

    X_all, y_all, subject_ids, class_labels = load_and_analyze_data(file_paths, class_labels)

    classification_pairs = ['40-80', 'P40-P80', '40-P40', '80-P80']
    all_results = []

    model_types = ['simple_lstm']

    print(f"\nModels to run: {model_types}")

    for pair_name in classification_pairs:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {pair_name}")
        print(f"{'=' * 60}")

        X_pair, y_pair, subject_ids_pair = create_binary_dataset(
            X_all, y_all, subject_ids, class_labels, pair_name
        )

        for model_type in model_types:
            print(f"\n{'=' * 40}")
            print(f"Model: {model_type.upper()}")
            print(f"{'=' * 40}")

            try:
                result = cross_validate_model(
                    X_pair, y_pair, subject_ids_pair,
                    pair_name, model_type, n_folds=5
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error training {model_type} for {pair_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_cv_results_to_excel(all_results, output_root)

    print_final_summary(all_results)

    if all_results:
        plot_overall_cv_comparison(all_results)

    return all_results


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("LSTM CLASSIFICATION WITH 5-FOLD CV")
        print("4:1 Subject-based Split")
        print("5-Fold Cross-Validation")
        print("=" * 60)

        results = main()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nAll results saved to: {output_root}/")
        print("   Subject-based 4:1 split")
        print("   5-Fold Cross-Validation")
        print("   Output ACC and F1 as mean ± std")
        print(f"   Models run: simple_lstm")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()