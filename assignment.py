#!/usr/bin/env python
# coding: utf-8

# # Prompt
# 
# You have graduated from this class, and are a huge success!
# You landed a job doing data science at some fancy company.
# 
# You just got a new client with some really interesting problems you get to solve.
# Unfortunately, because of a big mess-up on their side the data's metadata got corrupted
# (and the person that used to maintain the data just took a vow of silence and moved to a bog).
# 
# The only column you are sure about is the `label` column,
# which contains a numeric label for each row.
# Aside from that, the client does not know anything about the names, content, or even data types for each column.
# 
# Your task is to explore, clean, and analyze this data.
# You should have already received an email with the details on obtaining your unique data.
# Place it in the same directory as this notebook (and your `local_grader.py` script) and name it `data.txt`.
# 
# *I know this prompt may sound unrealistic, but I have literally been in a situation exactly like this.
# I was working at a database startup, and one of our clients gave us data with over 70 columns and more than a million records and told us:
# "The person who used to manage the data is no longer working with us, but this was the data they used to make all their decisions.
# We also lost all the metadata information, like column names."
# ...
# Working in industry is not always glamorous.
# -Eriq*

# # Part 0: Explore Your Data
# 
# Before you start doing things to/with your data, it's always a good idea to load up your data and take a look.

# In[1]:


import pandas
import numpy
import sklearn
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind

# Modify this to point to your data.
unique_data = pandas.read_csv('data.txt', sep = "\t")
unique_data


# Don't forget to checkout the column information.

# In[ ]:


unique_data.info()


# And any numeric information.

# In[ ]:


unique_data.describe()


# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Introduction</h4>
# 
# Briefly describe the dataset you’re given and define the goal of the project and how you approach it.
# For example, you can present a basic introduction of your data (shape and proposed data types)
# and your goal is to use these features to predict the label of the response variable.
# Then you propose a few models that are suitable for this project which will be introduced in the modeling section.

# # Part 1: Data Cleaning
# 
# As always, we should start with data cleaning.
# Take what you learned from HO3 to clean up this messy data to a point where it is ready for machine learning algorithms.
# 
# Some things you may want to do:
#  - Deal with missing/empty values.
#  - Fix numeric columns so that they actually contain numbers.
#  - Remove inconsistencies from columns.
#  - Assign a data type to each column.

# <h4 style="color: darkorange; font-size: x-large";>★ Task 1.A</h4>
# 
# Complete the following function that takes in a DataFrame and outputs a clean version of the DataFrame.
# You can assume that the frame has all the same structure as your unique dataset.
# You can return the same or a new data frame.

# In[ ]:


def clean_data(frame):
    # Removing rows with empty label column
    if 'label' in frame.columns:
        frame = frame.dropna(subset=['label'])

    # Making all empty values match in name
    frame.replace(['N/A', '?', 'NULL', 'n/a', 'None'], pandas.NA, inplace=True)

    # Cleaning string columns
    for i in frame.columns:
        if frame[i].dtype == object:
            frame[i] = frame[i].dropna().astype(str).str.strip().str.title()

    # Converting numeric strings to numeric types
    for i in frame.columns:
        if frame[i].dtype == object:
            try:
                if frame[i].str.contains(r'\d').any():
                    frame[i] = frame[i].apply(
                        lambda x: float(''.join(c for c in x if c.isdigit() or c == '.'))
                        if any(c.isdigit() for c in x) else x
                    )
            except Exception:
                pass

    # Assigning data types EXPLICITLY
    for i in frame.columns:
        if frame[i].dtype == object:
            frame[i] = frame[i].astype('object')
        elif frame[i].dtype == float:
            if frame[i].dropna().mod(1).eq(0).all():
                frame[i] = frame[i].astype('int64')
            else:
                frame[i] = frame[i].astype('float64')
        elif frame[i].dtype == int:
            frame[i] = frame[i].astype('int64')

    frame = frame.reset_index(drop=True)
    return frame

unique_data = clean_data(unique_data)
unique_data


# Now we should also be able to view all the numeric columns.

# In[ ]:


unique_data.info()


# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Data Cleaning</h4>
# 
# Describe the steps you took for data cleaning.
# Why did you do this?
# Did you have to make some choices along the way? If so, describe them.

# # Part 2: Data Visualization
# 
# Once you have cleaned up the data, it is time to explore it and find interesting things.
# Part of this exploration, will be visualizing the data in a way that makes it easier for yourself and others to understand.
# Use what you have learned in HO1 and HO2 to create some visualizations for your dataset.

# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Data Visualization</h4>
# 
# Create at least two different visualizations that help describe what you see in your dataset.
# Include these visualizations in your report along with descriptions of
# how you created the visualization,
# what data preparation you had to do for the visualization (aside from the data cleaning in the previous part),
# and what the visualization tells us about the data.

# # Part 3: Modeling
# 
# Now that you have a good grasp of your clean data,
# it is time to do some machine learning!
# (Technically all our previous steps were also machine learning,
# but now we get to use classifiers!)
# 
# Use the skills you developed to select **three** classifiers and implement them on your data.
# For example, you can narrow down your choices to three classifiers which may include:
# - Logistic regression
# - K-nearest neighbors
# - Decision tree
# - Or others

# <h4 style="color: darkorange; font-size: x-large";>★ Task 3.A</h4>
# 
# Complete the following function that takes in no parameters,
# and returns a list with **three** untrained classifiers you are going to explore in this assignment.
# This method may set parameters/options for the classifiers, but should not do any training/fitting.
# 
# For example, if you wanted to use logistic regression,
# then **one** of your list items may be:
# ```
# sklearn.linear_model.LogisticRegression()
# ```

# In[ ]:


def create_classifiers():
    logistic_regression = sklearn.linear_model.LogisticRegression()
    decision_tree = DecisionTreeClassifier()
    svc = sklearn.svm.SVC()
    return [logistic_regression, decision_tree, svc]

my_classifiers = create_classifiers()
my_classifiers


# Now that we have some classifiers, we can see how they perform.

# <h4 style="color: darkorange; font-size: x-large";>★ Task 3.B</h4>
# 
# Complete the following function that takes in an untrained classifier, a DataFrame, and a number of folds.
# This function should run k-fold cross validation with the classifier and the data,
# and return a list with the accuracy of each run of cross validation.
# You can assume that the frame has the column `label` and the rest of the columns can be considered clean numeric features.
# 
# Note that you may have to break your frame into features and labels to do this.
# Do not change the passed-in frame (make copies instead).
# 
# If you are getting any `ConvergenceWarning`s you may either ignore them,
# or try and address them
# (they will not affect your autograder score, but may be something to discuss in the written portion of this assignment).

# In[ ]:


def cross_fold_validation(classifier, frame, folds):
    X = frame.iloc[:, 1:]
    y = frame.iloc[:, 0]

    # Identifying each column
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Helper func for data preprocessing
    def preprocess_data(X):
        X[categorical_cols] = X[categorical_cols].fillna('Missing')

        # One hot encoding for categorical data
        one_hot_encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        X_categorical = one_hot_encoder.fit_transform(X[categorical_cols]).toarray()

        X_numeric = X[numeric_cols].to_numpy()

        # Scaling numeric data
        scaler = sklearn.preprocessing.StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)

        # Combining data
        return numpy.hstack([X_numeric, X_categorical])

    X_preprocessed = preprocess_data(X)

    scores = sklearn.model_selection.cross_val_score(
        classifier, X_preprocessed, y, cv=folds, scoring='accuracy'
    )
    return scores.tolist()

my_classifiers_scores = []
for classifier in my_classifiers:
    accuracy_scores = cross_fold_validation(classifier, unique_data, 5)
    my_classifiers_scores.append(accuracy_scores)
    print("Classifier: %s, Accuracy: %s." % (type(classifier).__name__, accuracy_scores))


# <h4 style="color: darkorange; font-size: x-large";>★ Task 3.C</h4>
# 
# Complete the following function that takes in two equally-sized lists of numbers and a p-value.
# This function should compute whether there is a statistical significance between
# these two lists of numbers using a [Student's t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)
# at the given p-value.
# Return `True` if there is a statistical significance, and `False` otherwise.
# Hint: If you wish, you may use the `ttest_ind()` [method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) provided in the scipy package. 

# In[ ]:


def significance_test(a_values, b_values, p_value):
    t_stat, p_val = ttest_ind(a_values, b_values, equal_var=False)
    return p_val < p_value

for i in range(len(my_classifiers)):
    for j in range(i + 1, len(my_classifiers)):
        significant = significance_test(my_classifiers_scores[i], my_classifiers_scores[j], 0.10)
        print("%s vs %s: %s" % (type(my_classifiers[i]).__name__,
                                type(my_classifiers[j]).__name__, significant))


# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Modeling</h4>
# 
# Describe the classifiers you have chosen.
# Be sure to include all details about any parameter settings used for the algorithms.
# 
# Compare the performance of your models using k-fold validation.
# You may look at accuracy, F1 or other measures.
# 
# Then, briefly summarize your results.
# Are your results statistically significant?
# Is there a clear winner?
# What do the standard deviations look like, and what do they tell us about the different models?
# Include a table like Table 1.
# 
# <center>Table 1: Every table needs a caption.</center>
# 
# | Model | Mean Accuracy | Standard Deviation of Accuracy |
# |-------|---------------|--------------------------------|
# | Logistic Regression | 0.724 | 0.004
# | K-Nearest Neighbor | 0.750 | 0.003
# | Decision Tree | 0.655 | 0.011

# # Part 4: Analysis
# 
# Now, take some time to go over your results for each classifier and try to make sense of them.
#  - Why do some classifiers work better than others?
#  - Would another evaluation metric work better than vanilla accuracy?
#  - Is there still a problem in the data that should fixed in data cleaning?
#  - Does the statistical significance between the different classifiers make sense?
#  - Are there parameters for the classifier that I can tweak to get better performance?

# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Analysis</h4>
# 
# Discuss your observations, the relationship you found, and how you applied concepts from the class to this project.
# For example, you may find that some feature has the most impact in predicting your response variable or removing a feature improves the model accuracy.
# Or you may observe that your training accuracy is much higher than your test accuracy and you may want to explain what issues may arise.

# # Part 5: Conclusion

# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: Conclusion</h4>
# 
# Briefly summarize the important results and conclusions presented in the project.
# What are the important points illustrated by your work?
# Are there any areas for further investigation or improvement?

# <h4 style="color: darkorange; font-size: x-large";>★ Written Task: References</h4>
# 
# Include a standard bibliography with citations referring to techniques or published papers you used throughout your report (if you used any).
# 
# For example:
# ```
# [1] Derpanopoulos, G. (n.d.). Bayesian Model Checking & Comparison.
# https://georgederpa.github.io/teaching/modelChecking.html.
# ```

# # Part XC: Extra Credit
# 
# So far you have used a synthetic dataset that was created just for you.
# But, data science is always more interesting when you are dealing with actual data from the real world.
# Therefore, you will have an opportunity for extra credit on this assignment using real-world data.
# 
# For extra credit, repeat the **written tasks** of Parts 0 through 4 with an additional dataset that you find yourself.
# For the written portion of the extra credit for Part 0, include information about where you got the data and what the data represents.
# You may choose any dataset that represents real data (i.e., is **not** synthetic or generated)
# and is **not** [pre-packaged in scikit-learn](https://scikit-learn.org/stable/datasets.html).
# 
# Below are some of the many places you can start looking for datasets:
#  - [Kaggle](https://www.kaggle.com/datasets) -- Kaggle is a website focused around machine learning competitions,
#        where people compete to see who can get the best results on a dataset.
#        It is very popular in the machine learning community and has thousands of datasets with descriptions.
#        Make sure to read the dataset's description, as Kaggle also has synthetic datasets.
#  - [data.gov](https://data.gov/) -- A portal for data from the US government.
#         The US government has a lot of data, and much of it has to be available to the public by law.
#         This portal contains some of the more organized data from several different government agencies.
#         In general, the government has A LOT of interesting data.
#         It may not always be clean (remember the CIA factbook), but it is interesting and available.
#         All data here should be real-world, but make sure to read the description to verify.
#  - [UCI's Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) -- UC Irvine has their own data repository with a few hundred datasets on many different topics.
#         Make sure to read the dataset's description, as UCI also has synthetic datasets.
#  - [WHO's Global Health Observatory](https://apps.who.int/gho/data/node.home) -- The World Health Organization keeps track of many different health-related statistics for most of the countries in the world.
#         All data here should be real-world, but make sure to read the description to verify.
#  - [Google's Dataset Search](https://datasetsearch.research.google.com/) -- Google indexes many datasets that can be searched here.
# 
# You can even create a dataset from scratch if you find some data you like that is not already organized into a specific dataset.
# The only real distinction between "data" and a "dataset" is that a dataset is organized and finite (has a fixed size).
# 
# Create a new section in your written report for this extra credit and include all the written tasks for the extra credit there.
# Each written task/section that you complete for your new dataset is eligible for extra credit (so you can still receive some extra credit even if you do not complete all parts).
# There is no need to submit any code for the extra credit.
# If you created a new dataset, include the dataset or links to it with your submission.
