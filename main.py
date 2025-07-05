from src.data_loader import load_data
from src.model import train_linear, train_tree
from src.evaluate import calculate_rmse, r2_score
from src.preprocessing import scale_features
from src.visuals import plot_predictions
from sklearn.model_selection import train_test_split

def main():
    df = load_data()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, return_scaler=False)

    # Linear
    linear_model = train_linear(X_train_scaled, y_train)
    y_pred_linear = linear_model.predict(X_test_scaled)
    rmse_linear = calculate_rmse(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Tree
    tree_model = train_tree(X_train_scaled, y_train)
    y_pred_tree = tree_model.predict(X_test_scaled)
    rmse_tree = calculate_rmse(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)

    # Output
    print(f"Linear Regression RMSE: {rmse_linear:.2f}")
    print(f"Linear Regression R2: {r2_linear:.2f}")
    print(f"Decision Tree RMSE: {rmse_tree:.2f}")
    print(f"Decision Tree R2: {r2_tree:.2f}")

    # Reusable plotting function used for both models — avoids duplicate code
    plot_predictions(y_test, y_pred_linear, model_name="Linear Regression")
    plot_predictions(y_test, y_pred_tree, model_name="Decision Tree")

if __name__ == "__main__":
    main()