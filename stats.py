import pandas as pd
from scipy.stats import chi2_contingency


df = pd.read_excel("dataset.xls")
df.fillna(0, inplace=True)

group_gender = df.groupby(["Gender", "ShortlistedNY"], dropna=False)["ApplicantCode"].count()
group_bame = df.groupby(["BAMEyn", "ShortlistedNY"], dropna=False)["ApplicantCode"].count()

# This calculates the relevant tables and values for the chi-squared hypothesis test
def chi_squared_p_value(df, group1, column, group2 = None):
    # This generates the table of values if 1 (Just Gender) or 2 (Both gender and ethnicity) are being checked
    if not group2:
        table = pd.crosstab(df[group1], column)
    else:
        table = pd.crosstab([df[group1], df[group2]], column)

    # This generates the relevant values for the test and  does not use the Yates correction.
    c, p, dof, expected = chi2_contingency(table, correction = False)

    return c, p, dof, expected, table


print(chi_squared_p_value(df, "BAMEyn", df["ShortlistedNY"]))
print(chi_squared_p_value(df, "Gender", df["ShortlistedNY"]))


def confusion_matrix_2_categories(X_test, y_test, y_predict, category, option_1, option_2):
    from sklearn.metrics import confusion_matrix
    from ml import split_data

    # splits testing data into privileged and unprivileged categories
    X_test_1, X_test_2 = split_data(X_test, category)


    # Producing confusion matricies for priv and unpriv group to compare
    # 1:
    # Generates list of applicantcodes that are in group 1
    X_test_1_id = X_test_1["ApplicantCode"].to_numpy()
    # All index values are 1 below the applicant code, so we subtract 1 from each
    X_test_1_index_value = X_test_1_id - 1
    # Remove duplicates
    X_test_1_index_value = list(dict.fromkeys(X_test_1_index_value))

    # 2:
    # Generates list of applicantcodes that are in group 2
    X_test_2_id = X_test_2["ApplicantCode"].to_numpy()
    # All index values are 1 below the applicant code, so we subtract 1 from each
    X_test_2_index_value = X_test_2_id - 1
    # Remove duplicates
    X_test_2_index_value = list(dict.fromkeys(X_test_2_index_value))


    # Splitting test and prediction results to group 1 and group 2 arrays
    y_test_1 = []
    y_test_2 = []
    y_predict_1 = []
    y_predict_2 = []
    for row, cat in enumerate(X_test[category]):
        if cat == option_1:
            y_predict_1.append(y_predict[row])
            y_test_1.append(y_test.to_numpy()[row])
        elif cat == option_2:
            y_predict_2.append(y_predict[row])
            y_test_2.append(y_test.to_numpy()[row])

    # Producing confusion matricies for overall, group 1 and group 2.
    print(f"Confusion Matricies: {category}")
    print("Total:")
    print(confusion_matrix(y_test, y_predict))

    print(f"{option_1}:")
    print(confusion_matrix(y_test_1, y_predict_1))

    print(f"{option_2}:")
    print(confusion_matrix(y_test_2, y_predict_2))

