from src.data_loader import load_data
from src.model import train_linear, train_tree
from src.evaluate import calculate_rmse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    df = load_data()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear
    linear_model = train_linear(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    rmse_linear = calculate_rmse(y_test, y_pred_linear)

    # Tree
    tree_model = train_tree(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    rmse_tree = calculate_rmse(y_test, y_pred_tree)

    # Output
    print(f"Linear Regression RMSE: {rmse_linear:.2f}")
    print(f"Decision Tree RMSE: {rmse_tree:.2f}")

    # Plot
    plt.scatter(y_test, y_pred_tree, alpha=0.3)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices (Tree)")
    plt.title("Tree Model: Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()