# ------------ Importaciones

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# ------------ 4. antiguedad "HiringDate" viene en formato m/d/aaaa, no esta estandarizada, con y sin ceros a la izquierda.

fechas_invalidas = EmpleadosAttrition["HiringDate"][
    pd.to_datetime(EmpleadosAttrition["HiringDate"], format="%m/%d/%Y", errors="coerce").isna()
]

print(fechas_invalidas.tolist())#hay una fecha que dice un dia que no existe, como solo se requiere saber la antiguedad en base al año, no se le hara nada

# ------------ 5. Crea una columna llamada Year y obtén el año de contratación del empleado a partir de su fecha ‘HiringDate’. No se te olvide que debe ser un entero.

EmpleadosAttrition["Year"] = (EmpleadosAttrition["HiringDate"].str.split("/").str[-1].astype(int))

# ------------ 6. Crea una columna llamada YearsAtCompany que contenga los años que el empleado lleva en la compañía hasta el año 2018. Para su cálculo, usa la variable Year que acabas de crear.

EmpleadosAttrition["YearsAtCompany"] = 2018 - EmpleadosAttrition["Year"]
EmpleadosAttrition[["HiringDate", "Year", "YearsAtCompany"]].head()

# ------------ 7. La DistanceFromHome está dada en kilómetros, pero tiene las letras “km” al final y así no puede ser entera.

EmpleadosAttrition = EmpleadosAttrition.rename(
    columns={"DistanceFromHome": "DistanceFromHome_km"}
)

# ------------ 8 y 9. Renombra la variable DistanceFromHome a DistanceFromHome_km.

EmpleadosAttrition["DistanceFromHome"] = (EmpleadosAttrition["DistanceFromHome_km"].str.replace(" km", "", regex=False).astype(int))

"""esto me garantiza que se puede conservar una columna donde aparezca la cantidad numerica y ademas especificar que son kilometros, y tambien una donde unicamente aparezca el dato numerico sin el string"""
EmpleadosAttrition[["DistanceFromHome_km", "DistanceFromHome"]].head()

# ------------ 10. Borra las columnas Year, HiringDate y DistanceFromHome_km debido a que ya no son útiles.

EmpleadosAttrition = EmpleadosAttrition.drop(columns=["Year", "HiringDate", "DistanceFromHome_km"])
EmpleadosAttrition.columns

# ------------ 11. Aprovechando los ajustes que se están haciendo, la empresa desea saber si todos los departamentos tienen un ingreso promedio similar. Genera una nuevo frame llamado SueldoPromedioDepto que contenga el MonthlyIncome promedio por departamento de los empleados y colócalo en una variable llamada SueldoPromedio. Esta tabla solo es informativa, no la vas a utilizar en el set de datos que estás construyendo.

SueldoPromedioDepto = (EmpleadosAttrition.groupby("Department")["MonthlyIncome"].mean().reset_index(name="SueldoPromedio"))

SueldoPromedioDepto

# ------------ 12. La variable MonthlyIncome tiene un valor numérico muy grande comparada con las otras variables. Escala dicha variable para que tenga un valor entre 0 y 1. 

scaler = MinMaxScaler()
EmpleadosAttrition["MonthlyIncome"] = scaler.fit_transform(
    EmpleadosAttrition[["MonthlyIncome"]]
)
EmpleadosAttrition["MonthlyIncome"].describe()

# ------------ 
# ------------ 
# ------------ 
# ------------ 
# ------------ 
# ------------ 
