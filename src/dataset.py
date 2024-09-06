# Este script carga datos desde fuentes externas y los guarda en la carpeta data/raw/.
# Ejemplo: Descargar un conjunto de datos de una API y guardarlo como CSV.
# Este script se encarga de la carga y guardado de datos desde y hacia diferentes fuentes.
# Incluye funciones para cargar datos crudos, transformarlos y guardarlos en la estructura del proyecto.

import pandas as pd
import os

class DataProcessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df = None

    def load_data(self):
        """
        Carga un archivo CSV desde la carpeta data/raw/.
        """
        self.df = pd.read_csv(self.raw_data_path)

    def clean_data(self):
        """
        Limpia el dataset cargado.
        """
        # Drop columns: 'Unnamed: 0', 'flight'
        self.df = self.df.drop(columns=['Unnamed: 0', 'flight'])
        # One-hot encode columns: 'airline', 'source_city' and 5 other columns
        for column in ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
            insert_loc = self.df.columns.get_loc(column)
            self.df = pd.concat([self.df.iloc[:, :insert_loc], pd.get_dummies(self.df.loc[:, [column]]), self.df.iloc[:, insert_loc+1:]], axis=1)
        # Scale columns 'duration', 'days_left' between 0 and 1
        new_min, new_max = 0, 1
        old_min, old_max = self.df['duration'].min(), self.df['duration'].max()
        self.df['duration'] = (self.df['duration'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        old_min, old_max = self.df['days_left'].min(), self.df['days_left'].max()
        self.df['days_left'] = (self.df['days_left'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    def save_data(self):
        """
        Guarda el dataset limpio en la carpeta data/processed/.
        """
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.df.to_csv(self.processed_data_path, index=False)

if __name__ == "__main__":
    processor = DataProcessor(raw_data_path=r'..\data\raw\Clean_Dataset.csv', processed_data_path=r'..\data\processed\Clean_Dataset_processed.csv')
    processor.load_data()
    processor.clean_data()
    processor.save_data()