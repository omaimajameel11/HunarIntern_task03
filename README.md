# HunarIntern_task03
Spam Email Detection using Machine Learning 

This project focuses on building a spam email detection system that can classify emails as either spam or ham (non-spam). The system is implemented using Python and scikit-learn, applying two machine learning algorithms: Naive Bayes and Support Vector Machines (SVM). The dataset used is a collection of labeled SMS/email messages where each message is tagged as spam or ham.

The preprocessing stage is a critical step to prepare the raw text data for analysis. Each message is first converted to lowercase, URLs are removed, punctuation and numbers are stripped, and the text is cleaned to retain only meaningful words. This ensures that the input to the model is consistent and free from noise. The target labels are then encoded numerically, with ham represented as 0 and spam represented as 1.

To transform the cleaned text into numerical features, the project uses TF-IDF Vectorization (Term Frequency–Inverse Document Frequency). This technique assigns weights to words based on their frequency in a message relative to their frequency across the entire dataset. It helps highlight the importance of specific words that are more discriminative in distinguishing between spam and ham messages.

Once the features are extracted, the dataset is split into training and testing subsets. Two models are trained separately: Multinomial Naive Bayes, which is efficient and commonly used for text classification tasks, and Linear Support Vector Classifier (SVM), which is powerful in handling high-dimensional text data. Both models are then evaluated on the test set.

For performance evaluation, metrics such as accuracy, precision, recall, and F1-score are reported using a classification report. The results show how well each model can detect spam messages while minimizing false positives and false negatives. Additionally, a confusion matrix can be generated to better visualize the distribution of predictions versus actual labels.

This project demonstrates the effectiveness of classical machine learning techniques in text classification. Naive Bayes performs well due to its probabilistic nature and efficiency on word-based features, while SVM provides strong accuracy and generalization. The comparison between the two helps highlight trade-offs between speed and accuracy.

Future improvements could include experimenting with advanced models such as Random Forests, Gradient Boosting, or deep learning approaches like LSTMs or Transformers for more complex datasets. Integrating the model into a real-world application, such as an email client or web service, would make the spam filter practical and user-friendly.
