# genre_classification

A musical genre is a category that brings together musical compositions that share criteria of affinity such as its function (dance music, religious, cinema), its instrumentation (vocal, instrumental, electronic), its social context or the content of its lyrics.

The task of gender classification consists of, given a song (audio or vector of latent factors) predicting the gender category. In this section we will cover songs based on audio. A classification system could be used to refine the cold start problem. If a person uploads a violin cover of a Rock song to YouTube, their gender can be predicted and this helps a future recommendation from another user with similar tastes.

When classifying by gender, we find several problems:

1. The subjectivity of categorization: There are cases where a song belongs to several genres. Each genre can, in turn, have many subgenres. For example, Rock can be subdivided into: alternative, folk, indie, hard, punk, among others.
2. Audio level, the variation of gender in the same song: For example the song Faith by George Michael. It begins with a church organ section, which could be confused with a religious or classical genre. The rest of the song, it is clear that it belongs to a mixture of Pop, Rock, Funk, among others.
3. There is a significant semantic gap between the audio signal and the genre.

My approach will be based on the content. We will use a subset of the dataset mentioned in the preliminary section, the [GTZAN](http://marsyasweb.appspot.com/download/data_sets/). It is a collection of 500 audios of songs of 30 seconds classified in 5 genres: Metal, classical, hip hop, country, reggae. We will take the first 9 seconds of each song, resulting in 388 frames of time.

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/melspectrograms_samples.jpg)

## Approach

1. We convert the audios to mel spectrograms: 1000x388x128 (songs x time x frequency)
2. We divide and mix the audios in train, test, validation (75%, 15%, 10%).
3. We train a CNN with the training set using the validation set.
4. We measure the accuracy with the testing set.
5. We present the results.

CNN consists of 2 intermediate convolutional layers of 128 filters each with a Pooling size of 3 and Dropout to add variation to the data and prevent overfitting. On top of this, we stacked a dense network of 128 neurons completely connected. The last layer will be a dense array of 5 neurons with softmax activation function representing the 5 genera to be predicted. The network was compiled with SGD as an optimizer and categorical cross entropy as a function of loss.

## Results

The network was trained during 27 seasons, giving an accuracy of 80% on the training set, 57% on the validation set and 70% on the testing set.

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/training_accuracy.png)

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/training_loss.png)

A confusion matrix is a tool that allows the visualization of the performance of an algorithm that is used in machine learning. Each column of the matrix represents the number of predictions of each class, while each row represents the instances in the real class. One of the benefits of confusion matrices is that they make it easy to see if the system is confusing two classes.

![alt text](https://github.com/francarranza/genre_classification/raw/master/report/confusion_matrix.png)

In our case we can see that for the genres: Country, metal and classical, the system classifies very well. Now, the system confuses the reggae genre with hip hop, achieving a reduced precision of 56% and 57% respectively. Finally, you can see some confusion between classical and country which is curious.
