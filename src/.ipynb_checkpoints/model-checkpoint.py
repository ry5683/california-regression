from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.preprocessing import StandardScaler

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_tree(X_train, y_train):
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)
    return tree