import pandas as pd
from env import user, password, host
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split


def get_zillow(user=user, password=password, host=host):
    """
    Acquire data from a SQL database of 2017 Zillow properties and cache it locally.

    Parameters:
    user (str): The username for accessing the MySQL database.
    password (str): The password for accessing the MySQL database.
    host (str): The address of the server where the Zillow database is hosted.

    Returns:
    pandas.DataFrame: A cleaned dataframe containing information on 2017 Zillow properties.
    """
    # Cached CSV
    filename = "zillow.csv"
    # If cached csv exists
    if os.path.isfile(filename):
        df = pd.read_csv(filename, dtype={"buildingclassdesc": "string"})
    # Pull from SQL
    else:
        df = pd.read_sql(
            """
                        SELECT *
                        FROM properties_2017
                        LEFT JOIN airconditioningtype using(airconditioningtypeid)
                        LEFT JOIN architecturalstyletype using(architecturalstyletypeid)
                        LEFT JOIN buildingclasstype using(buildingclasstypeid)
                        LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid)
                        LEFT JOIN predictions_2017 using(parcelid)
                        LEFT JOIN propertylandusetype using(propertylandusetypeid)
                        LEFT JOIN storytype using(storytypeid)
                        LEFT JOIN typeconstructiontype using(typeconstructiontypeid)
                        where transactiondate like %s
                        """,
            f"mysql+pymysql://{user}:{password}@{host}/zillow",
            params=[("2017%",)],
        )
        # cache data locally
        df.to_csv(filename, index=False)
    # sort by column: 'transactiondate' (descending) for dropping dupes keeping recent
    df = df.drop_duplicates(subset="parcelid", keep="last")
    # no null lat long
    # Single Family Homes
    single_fam = [
        "Single Family Residential",
        "Condominium",
        "Residential General",
        "Manufactured, Modular, Prefabricated Homes",
        "Mobile Home",
        "Townhouse",
    ]

    df = df[df["propertylandusedesc"].isin(single_fam)]
    df = df[df["latitude"].notna()]
    df = df[df["longitude"].notna()]
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    return df


def handle_missing_values(df, column_pct, row_pct):
    """
    Drop rows or columns based on the percent of values that are missing.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    prop_required_column (float): The proportion of non-missing values required to keep a column.
    prop_required_row (float): The proportion of non-missing values required to keep a row.

    Returns:
    pandas.DataFrame: The dataframe with missing values handled.
    """
    temp_df = df
    # Drop columns with too many missing values
    threshold = int(round(column_pct * len(df.index), 0))
    temp_df.dropna(axis=1, thresh=threshold, inplace=True)

    # Drop rows with too many missing values
    threshold = int(round(row_pct * len(df.columns), 0))
    temp_df.dropna(axis=0, thresh=threshold, inplace=True)

    return temp_df


def check_columns(df, reports=False, graphs=False):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, the proportion of null values,
    the data type of the column, and the range of the column if it is float or int. The resulting dataframe is sorted by the
    'Number of Unique Values' column in ascending order.

    Args:
    - df: pandas dataframe

    Returns:
    - pandas dataframe
    """
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    if reports == True:
        print(df.info)
        print(df.describe())
    if graphs == True:
        numeric = df.select_dtypes(exclude=["object", "category"]).columns.to_list()
        for col in numeric:
            fig, ax = plt.subplots(figsize=(8, 2))
            sns.histplot(df, x=col, ax=ax)
            ax.set_title(col)
            plt.show()
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, proportion of null values, and data type to the data list
        if df[column].dtype in ["float64", "int64"]:
            data.append(
                [
                    column,
                    df[column].dtype,
                    df[column].nunique(),
                    df[column].isna().sum(),
                    df[column].isna().mean().round(5),
                    df[column].unique(),
                    df[column].describe()[["min", "max", "mean"]].values,
                ]
            )
        else:
            data.append(
                [
                    column,
                    df[column].dtype,
                    df[column].nunique(),
                    df[column].isna().sum(),
                    df[column].isna().mean().round(5),
                    df[column].unique(),
                    None,
                ]
            )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', 'Proportion of Null Values', 'dtype', and 'Range' (if column is float or int)
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "col_name",
            "dtype",
            "num_unique",
            "num_null",
            "pct_null",
            "unique_values",
            "range (min, max, mean)",
        ],
    )


def handle_missing_values(df, column_pct, row_pct):
    """
    Drop rows or columns based on the percent of values that are missing.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    prop_required_column (float): The proportion of non-missing values required to keep a column.
    prop_required_row (float): The proportion of non-missing values required to keep a row.

    Returns:
    pandas.DataFrame: The dataframe with missing values handled.
    """
    temp_df = df
    # Drop columns with too many missing values
    threshold = int(round(column_pct * len(df.index), 0))
    temp_df.dropna(axis=1, thresh=threshold, inplace=True)

    # Drop rows with too many missing values
    threshold = int(round(row_pct * len(df.columns), 0))
    temp_df.dropna(axis=0, thresh=threshold, inplace=True)
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    return temp_df


def box_plotter(df):
    """
    Generates a box plot for all columns in a dataframe using matplotlib.
    """
    for col in df.columns:
        try:
            plt.figure(figsize=(12, 1))
            plt.boxplot(df[col], vert=False)
            plt.title(col)
            plt.show()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            print(
                f"Number of results in lower quartile: {len(df[df[col] < lower_bound])} ({(len(df[df[col] < lower_bound])/len(df))*100:.2f}%)"
            )
            print(
                f"Number of results in inner quartile: {len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])} ({(len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])/len(df))*100:.2f}%)"
            )
            print(
                f"Number of results in upper quartile: {len(df[df[col] > upper_bound])} ({(len(df[df[col] > upper_bound])/len(df))*100:.2f}%)"
            )
        except:
            print(
                f"Error: Could not generate box plot for column {col}. Skipping to next column..."
            )
            plt.close()
            continue


def encode_columns(df, columns, drop_first=True):
    """
    Encode the specified columns of a dataframe using pd.get_dummies.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    columns (list): A list of column names to encode.

    Returns:
    pandas.DataFrame: The encoded dataframe.
    """
    # Encode the specified columns using pd.get_dummies
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=drop_first)

    return df_encoded


def multi_scaler(train, val, test, scaled_features=None, scaler="MM"):
    """
    This function takes in 3 dataframes (train, val, test)
    and scales them using the specified scaler.

    Parameters:
    train (pandas.DataFrame): The training dataframe.
    val (pandas.DataFrame): The validation dataframe.
    test (pandas.DataFrame): The test dataframe.
    scaled_features (list): A list of column names to scale. If None, all object columns are scaled.
    scaler (str): The scaler to use. Must be one of "MM" (MinMaxScaler), "Standard" (StandardScaler), or "Robust" (RobustScaler).

    Returns:
    tuple: A tuple of the scaled dataframes (train_scaled, val_scaled, test_scaled).
    """
    if scaled_features is None:
        # If scaled_features is not defined, scale all numeric columns
        numeric_cols = train.select_dtypes(include=["number"]).columns.to_list()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns to scale.")
        scaled_features = numeric_cols

    if scaler == "MM":
        scaler_obj = MinMaxScaler()
    elif scaler == "Standard":
        scaler_obj = StandardScaler()
    elif scaler == "Robust":
        scaler_obj = RobustScaler()
    else:
        raise ValueError(
            "Invalid scaler. Must be one of 'MM', 'Standard', or 'Robust'."
        )

    # We fit/transform on itself to prevent leakage of one set into the other
    # Training
    train_scaled = train.copy()
    train_scaled[scaled_features] = scaler_obj.fit_transform(train[scaled_features])
    # Validation
    val_scaled = val.copy()
    val_scaled[scaled_features] = scaler_obj.transform(val[scaled_features])
    # Test
    test_scaled = test.copy()
    test_scaled[scaled_features] = scaler_obj.transform(test[scaled_features])

    return train_scaled, val_scaled, test_scaled


def split_data(df, random_state=123):
    """Split into train, validate, test with a 60% train, 20% validate, 20% test"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    print(f"train: {len(train)} ({round(len(train)/len(df)*100)}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df)*100)}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df)*100)}% of {len(df)})")
    return train, validate, test


