# Toy NLP project based on recipe data from Epicurious found on [Kaggle](https://www.kaggle.com/hugodarwood/epirecipes)

I trained a simple LSTM-based model to detect mentions of food in recipes as an example of Named Entity Recognition. Please see my blog post about this project [here.](http://www.hiddenlayercake.com)

## Training data
To make training data, I hand-labeled food mentions in a random subset of 300 recipes, consisting of 981 sentences, using the awesome labeling tool [Prodigy](https://prodi.gy).

## Model training
I used a simple LSTM-based architecture implemented in Tensorflow (training notebook [here](https://github.com/carolmanderson/food/blob/master/notebooks/modeling/Train_basic_LSTM_model.ipynb)).

## Results
The best model checkpoint achieved a weighted average precision and recall both equal to 98.7 on the dev set, though I think this performance is a bit of an overestimate due to a caveat in how I split the data (see the blog post for details).

You can play with the model in the interactive web app [here.](http://54.213.148.85:8501)


