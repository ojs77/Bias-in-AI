import pandas as pd
from stats import chi_squared_p_value


df = pd.read_excel("dataset.xls")

# Fill the blank spaces in the dataframe with zeros
df.fillna(0, inplace=True)

def split_data(data, category):
    print(data)
    df_1 = data[data[category] == 1]
    df_2 = data[data[category] == 2]

    return df_1, df_2

def extract_data(data, category):
    X = data.iloc[:, :5]
    y = data[category]
    return X, y

def pre_process_data(df):
    import numpy as np
    
    # Generates expected and observed values to be used to find weights
    c, p, dof, expected, observed = chi_squared_p_value(df, "Gender", df["OfferNY"])
    
    # Generates weights for each option and rounds them to the nearest whole number as necessary for the following step
    weights_matrix = expected/observed.to_numpy()
    weights_matrix = np.around(weights_matrix*100, decimals = 0)
    
    # Assigns weights to every row
    weights_column = []
    for index, row in df.iterrows():
        weight = weights_matrix[int(row["Gender"]-1)][int(row["OfferNY"])]
        weights_column.append(weight)
    
    df["Weight"] = weights_column

    # The resampled df, every row is repeated in proportion to its weight
    resampled_df = []
    for index, row in df.iterrows():
        resampled_df += [row for x in range(int(row["Weight"]))]

    # Assigning column names to the new df
    resampled_df = pd.DataFrame(resampled_df, columns = ['ApplicantCode', 'Gender', 'BAMEyn', 'ShortlistedNY', 'Interviewed',
       'FemaleONpanel', 'OfferNY', 'AcceptNY', 'JoinYN'])
    
    return resampled_df



# This runs the dataframe through the pre-processing stage and returns a new dataframe
resampled_df = pre_process_data(df)

# Splits df to X (Features which include ApplicantCode, Gender, BAMEyn, Shortlisted, Interviewed and FemaleOnPanel) and y(OfferNY)
X, y = extract_data(resampled_df, "OfferNY")


# Splits X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state= 1, stratify=y)


import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline using KNN:
def knn_pipeline(X_train, y_train): 
    from sklearn.neighbors import KNeighborsClassifier
    # Using KNearestNeighbours classifier where K, the number of neighbours used to classify,
    # is the square root of the number of rows of the dataset, as that is a general good optimum value:
    # Amey Band, How to find the optimal value of K in KNN?, towardsdatasci-ence.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb, 2020
    clf_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train)))))

    clf_KNN.fit(X_train, y_train)

    return clf_KNN


# Stats:
from stats import confusion_matrix_2_categories
from sklearn.metrics import classification_report


# Generates predictions on the test set to be used for stats
y_predict = knn_pipeline(X_train, y_train).predict(X_test)

print("Overall class report", classification_report(y_test, y_predict))

# Generates confusion matricies for each group for gender and ethnicity.
confusion_matrix_2_categories(X_test, y_test, y_predict, "Gender", 1, 2)
confusion_matrix_2_categories(X_test, y_test, y_predict, "BAMEyn", 1, 2)