#Plots de los datos resultantes del entrenameinto del modelo

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class Plotter:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.model = joblib.load(model_path)

    def plot_correlation(self):
        """
        Crea un heatmap de correlación de las variables numéricas del dataset.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_distribution(self):
        """
        Crea un gráfico de distribución de las variables numéricas del dataset.
        """
        self.df.hist(bins=20, figsize=(12, 10))
        plt.suptitle('Feature Distribution')
        plt.show()
        
    def plot_price_distribution(self):
        """
        Crea un gráfico de distribución de la variable objetivo 'price'.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df['price'], bins=20, kde=True)
        plt.title('Price Distribution')
        plt.show()
        
    def importance_features(self):
        """
        Crea un gráfico de barras con la importancia de las características en el modelo.
        """
        features = self.df.drop(columns=['price']).columns
        importances = self.model.feature_importances_
        indices = importances.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
        plt.title('Feature Importance')
        plt.show()
        
        
if __name__ == "__main__":
    plotter = Plotter(data_path=r'..\data\processed\Clean_Dataset_processed.csv', model_path=r'..\models\RandomForestRegressor.pkl')
    plotter.plot_correlation()
    plotter.plot_distribution()
    plotter.plot_price_distribution()
    plotter.importance_features()