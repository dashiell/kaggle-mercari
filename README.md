# kaggle-mercari

This is for a kaggle competition located at https://www.kaggle.com/c/mercari-price-suggestion-challenge

This uses a neural network for a regression problem, and involves several categorical features and some text features. For the text features, we learn the word embeddings, not use pre-trained embeddings. The main reason for this is that any cleaning of the text resulted in higher CV error (measured as root mean squared log error)

The data for this script is available at kaggle.  The easiest way to run this is to install anaconda, and packages for tensorflow and Keras.

Because this was for a "kernels" only contest is limited to running on kaggle only, with no GPU available, this approach doesn't work well for the competition; training time is too long for the kaggle 60 minute limit with 4 vCPUs.  However, you can achieve what would be a top ~100 score by training on any computer with a GPU in ~10 minutes.
