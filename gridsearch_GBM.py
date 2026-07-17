import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings
import os
import time

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
    all_data = []
    all_labels = []
    all_subject_ids = []

    for i, file_path in enumerate(file_paths):
        if os.path.exists(file_path):
            print(f"Loading: {file_path}")
            df = pd.read_csv(file_path)
            print(f"  Data shape: {df.shape}")

            for j in range(len(df)):
                all_data.append(df.iloc[j].values)
                all_labels.append(i)
                all_subject_ids.append(f"Class{i}_Subject{j // 3}")
        else:
            print(f"File not found: {file_path}")

    X = np.array(all_data)
    y = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, subject_ids


def create_subject_split(X, y, subject_ids, test_size=0.2):
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
        n_test = int(np.ceil(len(subjects) * test_size))
        np.random.shuffle(subjects)
        test_subjects.extend(subjects[:n_test])
        train_subjects.extend(subjects[n_test:])

    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

    print(f"\nTraining subjects: {len(train_subjects)}, Training samples: {len(train_indices)}")
    print(f"Test subjects: {len(test_subjects)}, Test samples: {len(test_indices)}")

    return train_indices, test_indices


def run_lgb_gridsearch():
    print("=" * 60)
    print("LIGHTGBM GRID SEARCH OPTIMIZATION")
    print("=" * 60)

    X, y, subject_ids = load_data()

    if len(X) == 0:
        print("Error: No data loaded!")
        return

    param_grid = {
        'num_leaves': [15, 31, 50, 80, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'n_estimators': [50, 100, 150, 200, 300]
    }

    total_combinations = (
            len(param_grid['num_leaves']) *
            len(param_grid['learning_rate']) *
            len(param_grid['n_estimators'])
    )

    print(f"\nGrid Search Parameters:")
    print(f"  num_leaves: {param_grid['num_leaves']}")
    print(f"  learning_rate: {param_grid['learning_rate']}")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  Total combinations: {total_combinations}")

    class_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]

    all_results = []

    for class1, class2 in class_pairs:
        print("\n" + "=" * 60)
        print(f"Processing: {class_labels[class1]} vs {class_labels[class2]}")
        print("=" * 60)

        mask = (y == class1) | (y == class2)
        X_binary = X[mask]
        y_binary = np.where(y[mask] == class1, 0, 1)
        subject_ids_binary = subject_ids[mask]

        count_class1 = np.sum(y_binary == 0)
        count_class2 = np.sum(y_binary == 1)

        print(f"Total samples: {len(X_binary)}")
        print(f"Class distribution: {class_labels[class1]}: {count_class1}, {class_labels[class2]}: {count_class2}")

        train_idx, test_idx = create_subject_split(X_binary, y_binary, subject_ids_binary)
        X_train, X_test = X_binary[train_idx], X_binary[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        print("\nStarting Grid Search...")
        print("This may take several minutes...")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=lgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        elapsed = time.time() - start_time
        print(f"\nGrid Search completed in {elapsed:.2f} seconds")

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"\nBest Parameters:")
        print(f"  num_leaves: {best_params['num_leaves']}")
        print(f"  learning_rate: {best_params['learning_rate']}")
        print(f"  n_estimators: {best_params['n_estimators']}")
        print(f"  Best CV F1: {best_score:.4f}")

        cv_results = grid_search.cv_results_
        sorted_indices = np.argsort(cv_results['mean_test_score'])[::-1]

        print("\nTop 5 parameter combinations:")
        print("-" * 70)
        print("{:<5} {:<15} {:<15} {:<15} {:<15}".format(
            'Rank', 'num_leaves', 'learning_rate', 'n_estimators', 'F1 Score'))
        print("-" * 70)

        for i, idx in enumerate(sorted_indices[:5]):
            leaves = cv_results['param_num_leaves'][idx]
            lr = cv_results['param_learning_rate'][idx]
            n_est = cv_results['param_n_estimators'][idx]
            mean_score = cv_results['mean_test_score'][idx]
            print("{:<5} {:<15} {:<15} {:<15} {:<15.4f}".format(
                i + 1, leaves, f'{lr:.4f}', n_est, mean_score))

        print("\nTraining final model with best parameters...")
        best_model = lgb.LGBMClassifier(
            **best_params,
            objective='binary',
            boosting_type='gbdt',
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='binary')

        y_proba = best_model.predict_proba(X_test)
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = 0.0

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        all_results.append({
            'class_pair': f"{class_labels[class1]}_vs_{class_labels[class2]}",
            'best_num_leaves': best_params['num_leaves'],
            'best_learning_rate': best_params['learning_rate'],
            'best_n_estimators': best_params['n_estimators'],
            'best_cv_f1': best_score,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'roc_auc': roc_auc
        })

    df = pd.DataFrame(all_results)
    os.makedirs('LightGBM_Results', exist_ok=True)
    df.to_csv('LightGBM_Results/lgb_gridsearch_results.csv', index=False)

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    print(f"\nResults saved to: LightGBM_Results/lgb_gridsearch_results.csv")
    print("=" * 60)
    print("LIGHTGBM GRID SEARCH COMPLETED!")

    return all_results


if __name__ == "__main__":
    try:
        import lightgbm as lgb

        print(f"LightGBM version: {lgb.__version__}")
    except ImportError:
        print("LightGBM not installed. Installing...")
        os.system("pip install lightgbm")

    results = run_lgb_gridsearch()