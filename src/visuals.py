import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, model_name="Model"):
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()