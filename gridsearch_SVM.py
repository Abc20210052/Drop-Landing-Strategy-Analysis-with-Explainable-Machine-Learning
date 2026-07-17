import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

file_paths = [
    r'F:\DL_40.csv',
    r'F:\DL_80.csv',
    r'F:\DL_P40.csv',
    r'F:\DL_P80.csv'
]
class_labels = ['40', '80', 'F40', 'F80']


def load_data():
    all_data, all_labels, all_subject_ids = [], [], []
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        for j in range(len(df)):
            subject_id = f"Class{i}_Subject{j // 3}"
            all_data.append(df.iloc[j].values)
            all_labels.append(i)
            all_subject_ids.append(subject_id)
    X = np.array(all_data)
    y = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, subject_ids


def create_cross_subject_split(X, y, subject_ids, test_size=0.2):
    unique_subjects = np.unique(subject_ids)
    subject_classes = {}
    for subj in unique_subjects:
        subj_indices = np.where(subject_ids == subj)[0]
        subj_class = y[subj_indices[0]]
        if subj_class not in subject_classes:
            subject_classes[subj_class] = []
        subject_classes[subj_class].append(subj)

    train_subjects, test_subjects = [], []
    for class_id, subjects in subject_classes.items():
        n_subjects = len(subjects)
        n_test = int(np.ceil(n_subjects * test_size))
        np.random.shuffle(subjects)
        test_subjects.extend(subjects[:n_test])
        train_subjects.extend(subjects[n_test:])

    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]
    return train_indices, test_indices


def run_svm_gridsearch():
    print("=" * 80)
    print("SVM CLASSIFICATION WITH GRID SEARCH")
    print("=" * 80)

    X, y, subject_ids = load_data()
    C_range = np.logspace(-3, 3, 7)

    class_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    all_results = []

    for class1, class2 in class_pairs:
        print(f"\nProcessing: {class_labels[class1]} vs {class_labels[class2]}")

        mask = (y == class1) | (y == class2)
        X_binary, y_binary = X[mask], np.where(y[mask] == class1, 0, 1)
        subject_ids_binary = subject_ids[mask]

        train_indices, test_indices = create_cross_subject_split(
            X_binary, y_binary, subject_ids_binary, test_size=0.2)

        X_train, X_test = X_binary[train_indices], X_binary[test_indices]
        y_train, y_test = y_binary[train_indices], y_binary[test_indices]

        grid_search = GridSearchCV(
            SVC(kernel='rbf', gamma='scale', random_state=42),
            param_grid={'C': C_range},
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_C = grid_search.best_params_['C']
        print(f"Best C: {best_C:.4f}, Best CV F1: {grid_search.best_score_:.4f}")

        best_model = SVC(kernel='rbf', C=best_C, gamma='scale',
                         probability=True, random_state=42)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='binary')

        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

        all_results.append({
            'class_pair': f"{class_labels[class1]}_vs_{class_labels[class2]}",
            'best_C': best_C,
            'test_acc': test_acc,
            'test_f1': test_f1
        })

    pd.DataFrame(all_results).to_csv('SVM_gridsearch_results.csv', index=False)
    print("\nResults saved to SVM_gridsearch_results.csv")
    return all_results


if __name__ == "__main__":
    run_svm_gridsearch()