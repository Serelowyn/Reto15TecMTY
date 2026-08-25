# ------------ Importaciones

import pandas as pd
import numpy as np

# ------------ Fin de las Importaciones

# ------------ 2.se lee el archivo del proyecto para crear un data frame

EmpleadosAttrition = pd.read_csv(r"C:\\Users\\sasor\\Desktop\\Tec de mty\\6. Machine learning\\1. Ingenieria de las caracteristicas\\proyecto\\RETO.csv")

"""me sirve para poder ver una forma general del dataset, para su limmpieza"""
EmpleadosAttrition.shape
EmpleadosAttrition.head()
EmpleadosAttrition.dtypes
EmpleadosAttrition.info()

# ------------ 3. se eliminan las columnas irrelevantes del df

columnas_irrelevantes = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
EmpleadosAttrition = EmpleadosAttrition.drop(columns=columnas_irrelevantes)

"""se confirma la eliminacion de las columnas que no se necesitan"""
EmpleadosAttrition.shape

# ------------ 


# ------------ 
# ------------ 
# ------------ 
# ------------ 
