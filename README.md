# Aplicación de Predicción de Precios de Casas

Este repositorio contiene una aplicación de predicción de precios de casas basada en modelos de aprendizaje automático. La aplicación permite a los usuarios ingresar datos sobre una casa y obtiene una estimación del precio de la vivienda utilizando un modelo de regresión.

## Archivos del Proyecto

El proyecto consta de tres archivos principales:

### 1. `endtoend.py`

Este archivo contiene el código de preparación de datos y entrenamiento de modelos. Aquí se realiza lo siguiente:

- Descarga y carga los datos de viviendas desde un enlace en línea.
- Divide los datos en conjuntos de entrenamiento y prueba.
- Realiza el preprocesamiento de datos, incluida la imputación de valores faltantes, la codificación de variables categóricas y la estandarización de características numéricas.
- Entrena varios modelos de regresión, incluyendo regresión lineal, árbol de decisiones y Random Forest.
- Evalúa los modelos utilizando validación cruzada y selecciona el mejor modelo.
- Guarda el modelo entrenado en un archivo para su uso posterior.

### 2. `predictive_system.py`

Este archivo contiene un script que utiliza el modelo entrenado para realizar predicciones de precios de viviendas basadas en los datos de entrada proporcionados. Aquí se realiza lo siguiente:

- Carga el modelo previamente entrenado desde el archivo.
- Procesa los datos de entrada proporcionados por el usuario para que coincidan con el formato utilizado durante el entrenamiento del modelo.
- Realiza una predicción del precio de la vivienda utilizando el modelo cargado.
- Muestra la predicción al usuario.

### 3. `web.py`

Este archivo es un script de aplicación web creado con Streamlit. Permite a los usuarios interactuar con el modelo de predicción de precios de casas de manera sencilla a través de una interfaz web. Los usuarios pueden ingresar información sobre una casa y recibir una predicción del precio de la vivienda en función de los datos proporcionados.

## Instrucciones de Uso

1. Clona este repositorio en tu máquina local.

2. Asegúrate de tener Python 3.x instalado.

3. Instala las dependencias necesarias
   
4. Ejecuta la aplicación web ejecutando el siguiente comando:

   ```bash
   streamlit run web.py
   ```

5. Abre el enlace proporcionado en tu navegador web y utiliza la aplicación para realizar predicciones de precios de viviendas.

¡Disfruta utilizando la aplicación de predicción de precios de viviendas!

**Nota:** Asegúrate de que el archivo `trained_model.sav` generado por `endtoend.py` esté presente en el mismo directorio que `predictive_system.py` y `web.py` para que la aplicación web pueda cargar el modelo entrenado.

Ejemplo de funcionamiento:
<img width="477" alt="Screenshot 2023-09-21 at 18 18 40" src="https://github.com/gozale/endtoend/assets/124909575/81a6c9ef-1140-42dc-8927-b7bd5a95501a">

