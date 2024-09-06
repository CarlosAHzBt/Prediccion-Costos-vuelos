# Este script entrena un modelo de Machine Learning y lo guarda en la carpeta models/.
# Ejemplo: Entrenamiento de un modelo de regresión lineal para predecir precios de viviendas.
# Este script entrena un modelo de Machine Learning utilizando los datos preprocesados.
# Incluye la separación de datos en conjuntos de entrenamiento y prueba, el entrenamiento del modelo, 
# y la evaluación del desempeño del modelo.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

class FlightCostPredictor:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = RandomForestRegressor()

    def load_data(self):
        df_clean = pd.read_csv(self.data_path)
        self.X, self.y = df_clean.drop(columns=['price']), df_clean['price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print(f'Score: {self.model.score(self.X_test, self.y_test)}')
        print(f'Mean Squared Error: {mean_squared_error(self.y_test, y_pred)}')
        print(f'Mean Absolute Error: {mean_absolute_error(self.y_test, y_pred)}')
        print(f'R2 Score: {r2_score(self.y_test, y_pred)}')

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f'Model saved at {self.model_path}')

if __name__ == "__main__":
    predictor = FlightCostPredictor(data_path=r'..\data\processed\Clean_Dataset_processed.csv', model_path=r'..\models\RandomForestRegressor.pkl')
    predictor.load_data()
    predictor.train_model()
    predictor.evaluate_model()
    predictor.save_model()