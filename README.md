# genre_classification

Un género musical es una categoría que reúne composiciones musicales que comparten criterios de afinidad tales como su función (música de danza, religiosa, cine), su instrumentación (vocal, instrumental, electrónica), su contexto social o el contenido de su letra. 

La tarea de clasificación de género consiste en, dada una canción (audio o vector de factores latentes) predecir la categoría de género. En esta sección abordaremos canciones basadas en audio. Un sistema de clasificación, podría utilizarse para afinar el problema de cold start. Si una persona sube a YouTube un cover de violín de una canción de Rock, se podrá predecir su género y que esto ayude a una futura recomendación de otro usuario con gustos similares.

A la hora de clasificar por género, nos encontramos con varios problemas:

1. La subjetividad de la categorización: Hay casos donde una canción pertenece a varios géneros. Cada género puede a su vez, tener muchos subgéneros. Por ejemplo el Rock puede subdividirse en: alternativo, folk, indie, hard, punk, entre otros.
2. A nivel audio, la variación de género en una misma canción: Por ejemplo la canción Faith de George Michael. Comienza con una sección de órgano de iglesia, lo cual podría confundirse con una género religioso o clásico. El resto de la canción, es claro que pertenece a una mezcla de Pop, Rock, Funk, entre otros.
3. Hay una brecha semántica importante entre la señal de audio y el género.

Nuestro enfoque será basado en el contenido. Utilizaremos un subset del dataset  mencionado en la sección preliminares, el GTZAN. Es una colección de 500 audios de canciones de 30 segundos clasificados en 5 géneros: Metal, classical, hip hop, country, reggae. Tomaremos los primeros 9 segundos de cada canción, resultando en 388 frames de tiempo.

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/melspectrograms_samples.jpg)

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
