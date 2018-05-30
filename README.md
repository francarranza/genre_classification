# genre_classification


Nuestro enfoque será basado en el contenido. Utilizaremos un subset del dataset  mencionado en la sección preliminares, el GTZAN. Es una colección de 500 audios de canciones de 30 segundos clasificados en 5 géneros: Metal, classical, hip hop, country, reggae. Tomaremos los primeros 9 segundos de cada canción, resultando en 388 frames de tiempo.


## Approach

1. Convertimos los audios a espectrogramas de mel: 1000x388x128 (canciones x tiempo x frecuencia)
2. Dividimos y mezclamos los audios en train, test, validation (75%, 15%, 10%).
3. Entrenamos una CNN con el training set usando el validation set.
4. Medimos la precisión con el testing set.
5. Presentamos los resultados.

La CNN consta de 2 capas convolutivas intermedias de 128 filtros cada una con un Pooling size de 3 y Dropout para agregar variación a los datos y prevenir overfitting. Encima de esto, apilamos una red densa de 128 neuronas completamente conectada. La última capa, será una densa de 5 neuronas con la función de activación softmax que representan los 5 géneros a predecir. La red fue compilada con SGD como optimizador y categorical cross entropy como función de loss.

## Resultados

La red fue entrenada durante 27 épocas, dando una precisión del 80% sobre el training set, 57% sobre el validation set y 70% sobre el testing set. 

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/training_accuracy.png)

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/training_loss.png)

Una matriz de confusión es una herramienta que permite la visualización del desempeño de un algoritmo que se emplea en aprendizaje automático. Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa a las instancias en la clase real. Uno de los beneficios de las matrices de confusión es que facilitan ver si el sistema está confundiendo dos clases.

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/confusion_matrix.png)

En nuestro caso podemos ver que para los géneros: Country, metal y classical, el sistema clasifica muy bien. Ahora, el sistema confunde bastante el género reggae con hip hop, logrando una precisión más reducida del 56% y 57% respectivamente. Por último, se puede ver algo de confusión entre classical y country lo cual es curioso.
