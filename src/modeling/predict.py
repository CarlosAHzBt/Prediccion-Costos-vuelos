import pandas as pd
import joblib

class FlightCostPredictor:
    def __init__(self, model_path, training_data_path):
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.model = self.load_model()
        self.reference_columns = self.load_training_data()

    def load_model(self):
        """
        Carga el modelo entrenado desde el archivo especificado.
        """
        return joblib.load(self.model_path)

    def load_training_data(self):
        """
        Carga los datos de entrenamiento para obtener las columnas de referencia.
        """
        df_train = pd.read_csv(self.training_data_path)
        return df_train.drop(columns=['price']).columns

    def clean_data(self, df):
        """
        Limpia el dataset cargado.
        """
        df = df.drop(columns='flight')

        # One-hot encode columns: 'airline', 'source_city' and 5 other columns
        for column in ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
            insert_loc = df.columns.get_loc(column)
            df = pd.concat([df.iloc[:, :insert_loc], pd.get_dummies(df.loc[:, [column]]), df.iloc[:, insert_loc+1:]], axis=1)
        # Scale columns 'duration', 'days_left' between 0 and 1
        new_min, new_max = 0, 1
        old_min, old_max = df['duration'].min(), df['duration'].max()
        df['duration'] = (df['duration'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        old_min, old_max = df['days_left'].min(), df['days_left'].max()
        df['days_left'] = (df['days_left'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        # Fill missing values with 0
        df = df.fillna(0)
        return df

    def align_columns(self, df):
        """
        Alinea las columnas de los nuevos datos con las columnas de referencia.
        """
        for col in self.reference_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.reference_columns]
        return df

    def predict(self, df):
        """
        Realiza predicciones sobre el conjunto de datos proporcionado.
        """
        df_clean = self.clean_data(df)
        df_aligned = self.align_columns(df_clean)
        predictions = self.model.predict(df_aligned)
        return predictions

# Crear una instancia de FlightCostPredictor
predictor = FlightCostPredictor(model_path=r'models\RandomForestRegressor.pkl', training_data_path=r'data\processed\Clean_Dataset_processed.csv')

# Cargar nuevos datos de un vuelo para predecir su costo en India
data_inventada_de_vuelo = {
    'airline': ['Indigo'],
    'flight': ['6E-657'],
    'source_city': ['Mumbai'],
    'departure_time': ['Early_Morning'],
    'stops': ['zero'],
    'arrival_time': ['Early_Morning'],
    'destination_city': ['Delhi'],
    'class': ['Business'],
    'duration': [190],
    'days_left': [1],
}

# Convertir los nuevos datos a un DataFrame
df_new = pd.DataFrame(data_inventada_de_vuelo)

# Realizar predicciones
predictions = predictor.predict(df_new)
print(f'Predicted cost of the flight in rupees: {predictions[0]}')