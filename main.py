from config import FOLDS
from train_fold import train_fold


if __name__ == "__main__":
    all_results = []

    for fold_name in FOLDS:
        result = train_fold(fold_name)
        all_results.append(result)

    print("\n===== TÜM FOLD EĞİTİMLERİ BİTTİ =====")
    for result in all_results:
        print(
            f"{result['fold_name']} -> "
            f"best_valid_loss: {result['best_valid_loss']:.4f} | "
            f"model: {result['model_path']}"
        )