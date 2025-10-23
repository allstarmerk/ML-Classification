import os
import random
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def encode_categoricals(df):
    """Encode categorical columns to numeric using LabelEncoder"""
    encoders = {}
    new_df = df.copy()
    for col in new_df.columns:
        if not pd.api.types.is_numeric_dtype(new_df[col]):
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col].astype(str))
            encoders[col] = le
    return new_df, encoders

def save_histogram(values, title, filename, bins=10):
    """Save histogram plot"""
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(values), color='r', linestyle='--', 
                label=f'Mean: {np.mean(values):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def save_lineplot(x_vals, y_vals, title, xlabel, ylabel, filename):
    """Save line plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def log_accuracies(name, accuracies):
    """Log accuracies to CSV with mean"""
    df = pd.DataFrame({"run": np.arange(1, len(accuracies)+1), "accuracy": accuracies})
    
    # Add mean as an additional row
    summary_data = {
        "run": ["", "MEAN"],
        "accuracy": ["", np.mean(accuracies)]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Combine data and summary
    final_df = pd.concat([df, summary_df], ignore_index=True)
    
    path = OUTPUT_DIR / f"{name}_accuracies.csv"
    final_df.to_csv(path, index=False)
    return path

# ============================================================================
# PART 1 — Decision Tree & Random Forest
# ============================================================================

def task1_convert_to_categorical():
    """
    Task 1: Convert numerical variables to categorical and save
    """
    print("\n" + "="*70)
    print("TASK 1: Converting Numerical Variables to Categorical")
    print("="*70)
    
    input_file = "heart-disease-classification.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing required file: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Create categorical variables with exact specifications from project
    df["Age_cat"] = pd.cut(df["Age"], 
                           bins=[0, 40, 55, float('inf')],
                           labels=['Young', 'Mid', 'Older'])
    
    df["RestingBP_cat"] = pd.cut(df["RestingBP"],
                                  bins=[0, 120, 140, float('inf')],
                                  labels=['Low', 'Normal', 'High'])
    
    df["Cholesterol_cat"] = pd.cut(df["Cholesterol"],
                                    bins=[0, 200, float('inf')],
                                    labels=['Normal', 'High'])
    
    df["MaxHR_cat"] = pd.cut(df["MaxHR"],
                             bins=[0, 140, 165, float('inf')],
                             labels=['Low', 'Normal', 'High'])
    
    # DROP the original numerical columns as per instructions
    df = df.drop(['Age', 'RestingBP', 'Cholesterol', 'MaxHR'], axis=1)
    
    output_file = "HeartDiseaseData.csv"
    df.to_csv(output_file, index=False)
    
    print(f"- Created categorical variables: Age_cat, RestingBP_cat, Cholesterol_cat, MaxHR_cat")
    print(f"- Dropped original numerical columns")
    print(f"- Saved to: {output_file}")
    print(f"- New dataset shape: {df.shape}")
    print(f"- New columns: {list(df.columns)}\n")
    
    return output_file

def task2_train_classifiers(n_runs=20):
    """
    Task 2: Train Decision Tree and Random Forest with different min_samples_split
    Uses min_samples_split=253 (at least 200) and min_samples_split=10
    """
    print("="*70)
    print("TASK 2: Training Decision Tree and Random Forest Classifiers")
    print("="*70)
    
    data_file = "HeartDiseaseData.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing required file: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Identify target column
    target_col = "HeartDisease" if "HeartDisease" in df.columns else df.columns[-1]
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical variables
    X_enc, _ = encode_categoricals(X)
    
    # Storage for results
    results = {
        'dt_253': [], 'rf_253': [],
        'dt_10': [], 'rf_10': []
    }
    
    print(f"\nRunning {n_runs} iterations with random train/test splits (75/25)...\n")
    
    # Run experiments
    for i in range(n_runs):
        # 75/25 split as specified
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.25, random_state=RANDOM_SEED + i
        )
        
        # min_samples_split = 253 (at least 200 as required)
        dt_253 = DecisionTreeClassifier(min_samples_split=253, random_state=RANDOM_SEED + i)
        dt_253.fit(X_train, y_train)
        results['dt_253'].append(accuracy_score(y_test, dt_253.predict(X_test)))
        
        rf_253 = RandomForestClassifier(n_estimators=250, min_samples_split=253, 
                                        random_state=RANDOM_SEED + i)
        rf_253.fit(X_train, y_train)
        results['rf_253'].append(accuracy_score(y_test, rf_253.predict(X_test)))
        
        # min_samples_split = 10
        dt_10 = DecisionTreeClassifier(min_samples_split=10, random_state=RANDOM_SEED + i)
        dt_10.fit(X_train, y_train)
        results['dt_10'].append(accuracy_score(y_test, dt_10.predict(X_test)))
        
        rf_10 = RandomForestClassifier(n_estimators=250, min_samples_split=10, 
                                       random_state=RANDOM_SEED + i)
        rf_10.fit(X_train, y_train)
        results['rf_10'].append(accuracy_score(y_test, rf_10.predict(X_test)))
    
    # Print results
    print("--- Results with min_samples_split = 253 ---")
    print(f"Decision Tree:  Mean = {np.mean(results['dt_253']):.4f}, Std = {np.std(results['dt_253']):.4f}")
    print(f"Random Forest:  Mean = {np.mean(results['rf_253']):.4f}, Std = {np.std(results['rf_253']):.4f}")
    
    print("\n--- Results with min_samples_split = 10 ---")
    print(f"Decision Tree:  Mean = {np.mean(results['dt_10']):.4f}, Std = {np.std(results['dt_10']):.4f}")
    print(f"Random Forest:  Mean = {np.mean(results['rf_10']):.4f}, Std = {np.std(results['rf_10']):.4f}")
    
    print("\n--- Analysis ---")
    dt_change = np.mean(results['dt_10']) - np.mean(results['dt_253'])
    rf_change = np.mean(results['rf_10']) - np.mean(results['rf_253'])
    print(f"Decision Tree accuracy change (10 vs 253): {dt_change:+.4f}")
    print(f"Random Forest accuracy change (10 vs 253): {rf_change:+.4f}")
    
    if abs(dt_change) > 0.02:
        print(f" Decision Tree shows SIGNIFICANT change when min_samples_split is reduced")
    else:
        print(f"- Decision Tree shows MINIMAL change when min_samples_split is reduced")
    
    if abs(rf_change) > 0.02:
        print(f" Random Forest shows SIGNIFICANT change when min_samples_split is reduced")
    else:
        print(f" Random Forest shows MINIMAL change when min_samples_split is reduced")
    
    print(f"\n Random Forest outperforms Decision Tree by: "
          f"{np.mean(results['rf_10']) - np.mean(results['dt_10']):.4f} (with min_samples=10)")
    
    # Save results
    log_accuracies("dt_minsplit253", results['dt_253'])
    log_accuracies("rf_minsplit253", results['rf_253'])
    log_accuracies("dt_minsplit10", results['dt_10'])
    log_accuracies("rf_minsplit10", results['rf_10'])
    
    print(f" Saved accuracy logs to outputs/\n")
    
    return results

# ============================================================================
# PART 2 — Multiple Classifiers on Diabetes Data
# ============================================================================

def task3_multiple_classifiers(n_runs=20):
    """
    Task 3: Train kNN, Naive Bayes, Logistic Regression, and SVM
    """
    print("="*70)
    print("TASK 3: Multiple Classifiers on Diabetes Dataset")
    print("="*70)
    
    data_file = "DiabetesData.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing required file: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"Dataset shape: {df.shape}")
    
    # Identify target column
    target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encode if needed
    X_enc, _ = encode_categoricals(X)
    
    # Storage for results
    knn_results = {k: [] for k in range(3, 21)}
    nb_gaussian = []
    nb_multinomial = []
    lr_results = []
    svm_results = []
    
    print(f"\nRunning {n_runs} iterations with 75/25 train/test split...\n")
    
    for i in range(n_runs):
        # 75/25 split as specified (test_size=0.25)
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.25, random_state=RANDOM_SEED + i
        )
        
        # Normalize for distance-based methods
        X_train_mean = X_train.mean()
        X_train_std = X_train.std() + 1e-8
        X_train_norm = (X_train - X_train_mean) / X_train_std
        X_test_norm = (X_test - X_train_mean) / X_train_std
        
        # kNN with k from 3 to 20
        for k in range(3, 21):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_norm, y_train)
            acc = accuracy_score(y_test, knn.predict(X_test_norm))
            knn_results[k].append(acc)
        
        # Naive Bayes - Gaussian
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        nb_gaussian.append(accuracy_score(y_test, gnb.predict(X_test)))
        
        # Naive Bayes - Multinomial (requires non-negative features)
        X_train_pos = X_train - X_train.min() + 1
        X_test_pos = X_test - X_train.min() + 1
        mnb = MultinomialNB()
        mnb.fit(X_train_pos, y_train)
        nb_multinomial.append(accuracy_score(y_test, mnb.predict(X_test_pos)))
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED + i)
        lr.fit(X_train_norm, y_train)
        lr_results.append(accuracy_score(y_test, lr.predict(X_test_norm)))
        
        # SVM with RBF kernel
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_SEED + i)
        svm.fit(X_train_norm, y_train)
        svm_results.append(accuracy_score(y_test, svm.predict(X_test_norm)))
    
    # Print results
    print("--- kNN Results (Average Test Accuracy for each k) ---")
    knn_avg = []
    for k in range(3, 21):
        avg_acc = np.mean(knn_results[k])
        knn_avg.append(avg_acc)
        print(f"k = {k:2d}: {avg_acc:.4f}")
    
    print(f"\n--- Naive Bayes Results ---")
    print(f"Gaussian Distribution:    {np.mean(nb_gaussian):.4f}")
    print(f"Multinomial Distribution: {np.mean(nb_multinomial):.4f}")
    
    print(f"\n--- Logistic Regression ---")
    print(f"Average Test Accuracy: {np.mean(lr_results):.4f}")
    
    print(f"\n--- SVM (RBF kernel, C=1.0, gamma='scale') ---")
    print(f"Average Test Accuracy: {np.mean(svm_results):.4f}")
    
    # Plot kNN vs k
    save_lineplot(range(3, 21), knn_avg, 
                  "kNN Performance vs. k Value", 
                  "k (Number of Neighbors)", 
                  "Average Test Accuracy", 
                  "knn_vs_k.png")
    print(f"\n Saved kNN plot to outputs/knn_vs_k.png")
    
    # Save results
    log_accuracies("nb_gaussian", nb_gaussian)
    log_accuracies("nb_multinomial", nb_multinomial)
    log_accuracies("logreg", lr_results)
    log_accuracies("svm", svm_results)
    print(f" Saved accuracy logs to outputs/\n")
    
    return {
        'knn': knn_results,
        'nb_gauss': nb_gaussian,
        'nb_multi': nb_multinomial,
        'lr': lr_results,
        'svm': svm_results
    }

def task4_feature_selection(m_values=[3, 7], n_models=100):
    """
    Task 4: Random feature selection experiment
    """
    print("="*70)
    print("TASK 4: Feature Selection with Random Subsets")
    print("="*70)
    
    data_file = "DiabetesData.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing required file: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Identify target
    target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encode if needed
    X_enc, _ = encode_categoricals(X)
    feature_names = list(X_enc.columns)
    n_features = len(feature_names)
    
    print(f"Total features available: {n_features}")
    print(f"Testing with m = {m_values}\n")
    
    # Single train/test split for this task
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.25, random_state=RANDOM_SEED
    )
    
    for m in m_values:
        print(f"\n{'='*70}")
        print(f"Testing with m = {m} features")
        print(f"{'='*70}")
        
        results = {
            'knn': [],
            'nb_gauss': [],
            'nb_multi': [],
            'lr': [],
            'svm': []
        }
        
        best_acc = {
            'knn': (0, None),
            'nb_gauss': (0, None),
            'nb_multi': (0, None),
            'lr': (0, None),
            'svm': (0, None)
        }
        
        for i in range(n_models):
            # Randomly select m features
            selected_indices = np.random.choice(n_features, m, replace=False)
            selected_features = [feature_names[idx] for idx in selected_indices]
            
            X_train_sub = X_train.iloc[:, selected_indices]
            X_test_sub = X_test.iloc[:, selected_indices]
            
            # Normalize
            mean = X_train_sub.mean()
            std = X_train_sub.std() + 1e-8
            X_train_norm = (X_train_sub - mean) / std
            X_test_norm = (X_test_sub - mean) / std
            
            # kNN with k=11
            knn = KNeighborsClassifier(n_neighbors=11)
            knn.fit(X_train_norm, y_train)
            acc = accuracy_score(y_test, knn.predict(X_test_norm))
            results['knn'].append(acc)
            if acc > best_acc['knn'][0]:
                best_acc['knn'] = (acc, selected_features)
            
            # Naive Bayes - Gaussian
            gnb = GaussianNB()
            gnb.fit(X_train_sub, y_train)
            acc = accuracy_score(y_test, gnb.predict(X_test_sub))
            results['nb_gauss'].append(acc)
            if acc > best_acc['nb_gauss'][0]:
                best_acc['nb_gauss'] = (acc, selected_features)
            
            # Naive Bayes - Multinomial
            X_train_pos = X_train_sub - X_train_sub.min() + 1
            X_test_pos = X_test_sub - X_train_sub.min() + 1
            mnb = MultinomialNB()
            mnb.fit(X_train_pos, y_train)
            acc = accuracy_score(y_test, mnb.predict(X_test_pos))
            results['nb_multi'].append(acc)
            if acc > best_acc['nb_multi'][0]:
                best_acc['nb_multi'] = (acc, selected_features)
            
            # Logistic Regression
            lr = LogisticRegression(max_iter=2000, random_state=i)
            lr.fit(X_train_norm, y_train)
            acc = accuracy_score(y_test, lr.predict(X_test_norm))
            results['lr'].append(acc)
            if acc > best_acc['lr'][0]:
                best_acc['lr'] = (acc, selected_features)
            
            # SVM
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=i)
            svm.fit(X_train_norm, y_train)
            acc = accuracy_score(y_test, svm.predict(X_test_norm))
            results['svm'].append(acc)
            if acc > best_acc['svm'][0]:
                best_acc['svm'] = (acc, selected_features)
        
        # Print summary
        print(f"\n--- Results Summary (m = {m}) ---")
        print(f"{'Classifier':<15} {'Max Acc':<10} {'Mean Acc':<10} {'Std':<10}")
        print("-" * 50)
        for clf_name in results.keys():
            max_acc = best_acc[clf_name][0]
            mean_acc = np.mean(results[clf_name])
            std_acc = np.std(results[clf_name])
            print(f"{clf_name:<15} {max_acc:<10.4f} {mean_acc:<10.4f} {std_acc:<10.4f}")
        
        print(f"\n--- Best Feature Subsets (m = {m}) ---")
        for clf_name, (acc, features) in best_acc.items():
            print(f"{clf_name}: {acc:.4f} - {features}")
        
        # Create histogram for each classifier
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        clf_names = ['knn', 'nb_gauss', 'nb_multi', 'lr', 'svm']
        clf_titles = ['kNN (k=11)', 'Naive Bayes (Gaussian)', 
                      'Naive Bayes (Multinomial)', 'Logistic Regression', 'SVM']
        
        for idx, (clf, title) in enumerate(zip(clf_names, clf_titles)):
            axes[idx].hist(results[clf], bins=10, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Test Accuracy')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{title} (m={m})')
            axes[idx].axvline(np.mean(results[clf]), color='r', 
                            linestyle='--', label=f'Mean: {np.mean(results[clf]):.4f}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'feature_selection_m{m}_histogram.png', dpi=300)
        print(f"\n Histogram saved to outputs/feature_selection_m{m}_histogram.png")
        
        # Save best features
        for clf_name, (acc, features) in best_acc.items():
            df_best = pd.DataFrame({'feature': features, 'accuracy': [acc] * len(features)})
            df_best.to_csv(OUTPUT_DIR / f'best_features_m{m}_{clf_name}.csv', index=False)
        
        print(f" Best feature subsets saved to outputs/\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MSML603 PROJECT 1: CLASSIFICATION")
    print("="*70 + "\n")
    
    # PART 1
    task1_convert_to_categorical()
    task2_train_classifiers(n_runs=20)
    
    # PART 2
    task3_multiple_classifiers(n_runs=20)
    task4_feature_selection(m_values=[3, 7], n_models=100)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETE!")
    print("="*70)
    print("Check 'outputs/' folder for:")
    print("  - Accuracy logs (CSV files)")
    print("  - Plots (PNG files)")
    print("  - Best feature subsets (CSV files)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()