import os
import itertools
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from env import user, password, host
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.metrics import mean_squared_error, r2_score


def get_zillow(user=user, password=password, host=host):
    """
    This function acquires data from a SQL database of 2017 Zillow properties and caches it locally.

    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `get_zillow` is returning a pandas DataFrame containing information on single family residential properties
    """
    # name of cached csv
    filename = "zillow.csv"
    # if cached data exist
    if os.path.isfile(filename):
        # read data from cached csv
        df = pd.read_csv(filename)
        # Print size
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")
    # wrangle from sql db if not cached
    else:
        # read sql query into df
        # 261 is single family residential id
        df = pd.read_sql(
            """select * 
                            from properties_2017 
                            left join predictions_2017 using(parcelid) 
                            where propertylandusetypeid in (261,279)""",
            f"mysql+pymysql://{user}:{password}@{host}/zillow",
        )
        # filter to just 2017 transactions
        df = df[df["transactiondate"].str.startswith("2017", na=False)]
        # cache data locally
        df.to_csv(filename, index=False)
        # print total rows and columns
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")
    return df


def check_columns(df_telco):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, the proportion of null values,
    and the data type of the column. The resulting dataframe is sorted by the
    'Number of Unique Values' column in ascending order.

    Args:
    - df_telco: pandas dataframe

    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df_telco.columns:
        # Append the column name, number of unique values, unique values, number of null values, proportion of null values, and data type to the data list
        data.append(
            [
                column,
                df_telco[column].nunique(),
                df_telco[column].unique(),
                df_telco[column].isna().sum(),
                df_telco[column].isna().mean(),
                df_telco[column].dtype,
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', 'Proportion of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype",
        ],
    )


def explore_prep_zillow(df):
    """
    Prepare uncleaned Zillow dataframe for exploration.

    Args:
        df (pandas.DataFrame): uncleaned Zillow dataframe

    Returns:
        pandas.DataFrame: cleaned Zillow dataframe
    """

    # Potential future feature engineering with more time. Columns dropped in the end for now.
    # Replace missing values with appropriate values or 0 where it makes sense
    df = df.fillna(
        {
            "numberofstories": 0,
            "fireplaceflag": 0,
            "yardbuildingsqft26": 0,
            "yardbuildingsqft17": 0,
            "unitcnt": 0,
            "threequarterbathnbr": 0,
            "pooltypeid7": 0,
            "pooltypeid2": 0,
            "pooltypeid10": 0,
            "poolsizesum": 0,
            "poolcnt": 0,
            "hashottuborspa": 0,
            "garagetotalsqft": 0,
            "garagecarcnt": 0,
            "fireplacecnt": 0,
            "lotsizesquarefeet": df["calculatedfinishedsquarefeet"],
        }
    )
    # Potential future feature engineering with more time. Columns dropped in the end for now.
    # Split transaction date to year, month, and day
    df_split = df["transactiondate"].str.split(pat="-", expand=True).add_prefix("trx_")
    df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)

    # Rename columns
    df = df.rename(
        columns=(
            {
                "yearbuilt": "year",
                "bedroomcnt": "beds",
                "bathroomcnt": "baths",
                "calculatedfinishedsquarefeet": "area",
                "taxvaluedollarcnt": "prop_value",
                "fips": "county",
                "trx_1": "trx_month",
                "trx_2": "trx_day",
                "numberofstories": "stories",
                "poolcnt": "pools",
            }
        )
    )

    # Filter out/drop columns that have too many nulls, are related to target, are dupes, or have no use for exploration or modeling
    df = df.drop(
        columns=[
            "id",
            "airconditioningtypeid",
            "architecturalstyletypeid",
            "basementsqft",
            "buildingclasstypeid",
            "buildingqualitytypeid",
            "calculatedbathnbr",
            "decktypeid",
            "finishedfloor1squarefeet",
            "finishedsquarefeet12",
            "finishedsquarefeet13",
            "finishedsquarefeet15",
            "finishedsquarefeet50",
            "finishedsquarefeet6",
            "fullbathcnt",
            "heatingorsystemtypeid",
            "lotsizesquarefeet",
            "pooltypeid10",
            "pooltypeid2",
            "pooltypeid7",
            "propertycountylandusecode",
            "propertylandusetypeid",
            "propertyzoningdesc",
            "rawcensustractandblock",
            "regionidcity",
            "regionidcounty",
            "regionidneighborhood",
            "regionidzip",
            "storytypeid",
            "threequarterbathnbr",
            "typeconstructiontypeid",
            "yardbuildingsqft17",
            "yardbuildingsqft26",
            "structuretaxvaluedollarcnt",
            "assessmentyear",
            "landtaxvaluedollarcnt",
            "taxamount",
            "taxdelinquencyflag",
            "taxdelinquencyyear",
            "censustractandblock",
            "id.1",
            "logerror",
        ]
    )

    # Drop nulls
    df = df.dropna()

    # Map county to fips
    df.county = df.county.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})

    # Convert certain columns to int
    ints = ["year", "beds", "area", "prop_value", "trx_month", "trx_day"]
    for i in ints:
        df[i] = df[i].astype(int)

    # Potential future feature engineering with more time. Columns dropped in the end for now.
    # Sort by column: 'transactiondate' (descending) for dropping dupes keeping recent
    df = df.sort_values(["transactiondate"], ascending=[False])

    # Drop duplicate rows in column: 'parcelid', keeping max trx date
    df = df.drop_duplicates(subset=["parcelid"])

    # Added age of the house (in 2017)
    df = df.assign(age=2017 - df.year)

    # Get 1% and 99% quantiles of the target column (could potentially be tuned further with more time)
    q1 = df.prop_value.quantile(0.01)
    q99 = df.prop_value.quantile(0.99)

    # Filter out the outliers
    df = df[(df.prop_value >= q1) & (df.prop_value <= q99)]
    print(f"Final Size")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}\n")
    return df


def plot_histograms(df):
    """
    This function plots histograms for the columns "baths", "beds", "county", "area", "prop_value", and "age" in the given dataframe.

    :param df: The dataframe to plot histograms for
    :return: None
    """
    for col in ["baths", "beds", "county"]:
        plt.hist(df[col])
        plt.title(col)
        plt.show()

    # Make the histograms bins large enough for bigger values
    for col in ["area", "prop_value", "age"]:
        plt.hist(df[col], 100)
        plt.title(col)
        plt.show()


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


def kruskal(s1, s2, s3):
    """A function take takes in 3 variables, and performs the kruskal
    wallis stats testing on the 3."""
    stat, p = stats.kruskal(s1, s2, s3)
    print("Kruskal-Wallis H-Test:\n", f"stat = {stat}, p = {p}")


def plot_property_value_by_county(df):
    """
    This function creates a scatter plot of property values by county using the given dataframe.

    :param df: The dataframe to use for plotting
    :return: None
    """
    # Split the data by county
    la_county = df[df.county == "LA"]["prop_value"]
    orange_county = df[df.county == "Orange"]["prop_value"]
    ventura_county = df[df.county == "Ventura"]["prop_value"]

    # Perform Kruskal-Wallis H-Test on prop_value for the 3 counties
    stat, p = stats.kruskal(la_county, orange_county, ventura_county)
    print("Kruskal-Wallis H-Test:\n", f"stat = {stat}, p = {p}")

    # Create a copy of the dataframe and scale the latitude and longitude labels
    p = df.copy()
    p = p.assign(lat=p.latitude / 1000000)
    p = p.assign(long=p.longitude / 1000000)
    p = p.sort_values("prop_value")

    # Set the size of the plot
    plt.figure(figsize=[12, 6])

    # Create a generic legend example
    sns.scatterplot(
        data=p, y="lat", x="long", hue="prop_value", palette="Greys", alpha=1
    )

    # Create a scatter plot for each county, with hue on property value
    sns.scatterplot(
        data=p[p.county == "LA"], y="lat", x="long", hue="prop_value", palette="Reds"
    )
    sns.scatterplot(
        data=p[p.county == "Orange"],
        y="lat",
        x="long",
        hue="prop_value",
        palette="Greens",
    )
    sns.scatterplot(
        data=p[p.county == "Ventura"],
        y="lat",
        x="long",
        hue="prop_value",
        palette="Blues",
    )

    # Add county labels to the plot
    plt.text(y=34, x=-119.25, s="Ventura County", fontsize=12, color="darkblue")
    plt.text(y=33.9, x=-118.7, s="LA County", fontsize=12, color="darkred")
    plt.text(y=33.5, x=-118.15, s="Orange County", fontsize=12, color="darkgreen")

    # Add a generic legend to the plot
    plt.legend(
        title="Prop Value",
        labels=["$400k", "$800k", "$1.2m", "$1.6m", "$2.0m"],
        frameon=False,
    )

    # Add a title to the plot and hide the x and y axis labels
    plt.title("Property Value based on County")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # Show the plot
    plt.show()


def plot_correlations(df):
    """
    This function creates a bar plot of the correlations between each column in the given dataframe.

    :param df: The dataframe to use for plotting
    :return: None
    """
    # Get a list of all columns in the dataframe
    cols = df.columns.to_list()

    # Loop through each column and create a bar plot of its correlations with all other columns
    for col in cols:
        # Calculate the correlations between the current column and all other columns
        corr = df[cols].corr()[col]
        # Sort the correlations in descending order and create a bar plot
        corr.sort_values(ascending=False).plot(kind="bar")
        # Set the title of the plot to the current column name
        plt.title(col)
        # Show the plot
        plt.show()


def plot_lmplot(df, target):
    """
    Plot linear regression between each feature and target variable.

    Args:
        df (pandas.DataFrame): dataframe containing features and target variable
        target (str): name of target variable

    Returns:
        None
    """
    for i in df.drop(columns=target):
        sns.lmplot(x=i, y=target, data=df, line_kws={"color": "orange"})
        plt.show()


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


def spearman(train, x, y, alt_hyp="two-sided"):
    """
    Calculate the Spearman's rank correlation coefficient and p-value between two variables,
    and plot a regression line using Seaborn.

    Parameters:
    train (pandas.DataFrame): The training dataset.
    x (str): The name of the first variable.
    y (str): The name of the second variable.
    alt_hyp (str, optional): The alternative hypothesis for the test. Default is "two-sided".

    Returns:
    None
    """
    r, p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f"R = {r}, P = {p}\n")
    sns.regplot(data=train, x=x, y=y, marker=".", line_kws={"color": "orange"})
    plt.show()


def std_zillow(train, validate, test, scale=None):
    """
    Scale the numerical features of the train, validate, and test datasets using the Standard Scaler method.

    Args:
        train (pandas.DataFrame): training dataset
        validate (pandas.DataFrame): validation dataset
        test (pandas.DataFrame): test dataset
        scale (list): list of columns to scale (default is None, which scales all numerical columns)

    Returns:
        tuple: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled test data)
    """
    # If scale is None, scale all numerical columns
    if scale is None:
        scale = train.select_dtypes(include=["float64", "int64"]).columns.to_list()

    # Create a StandardScaler object
    std_scale = StandardScaler()

    # Create copies of the train, validate, and test dataframes
    Xtr, Xv, Xt = train.copy(), validate.copy(), test.copy()

    # Add "_std" to the column names of the columns to be scaled
    scale_std = [col + "_std" for col in scale]

    # Scale the columns in the train, validate, and test dataframes
    Xtr[scale_std] = std_scale.fit_transform(train[scale])
    Xv[scale_std] = std_scale.transform(validate[scale])
    Xt[scale_std] = std_scale.transform(test[scale])

    # Drop the unscaled columns from the train, validate, and test dataframes
    Xtr.drop(columns=scale, inplace=True)
    Xv.drop(columns=scale, inplace=True)
    Xt.drop(columns=scale, inplace=True)

    return Xtr, Xv, Xt


def mm_zillow(train, validate, test, scale=None):
    """
    Scale the numerical features of the train, validate, and test datasets using the MinMax Scaler method.

    Args:
        train (pandas.DataFrame): training dataset
        validate (pandas.DataFrame): validation dataset
        test (pandas.DataFrame): test dataset
        scale (list): list of columns to scale (default is None, which scales all numerical columns)

    Returns:
        tuple: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled test data)
    """
    # If scale is None, scale all numerical columns
    if scale is None:
        scale = train.select_dtypes(include=["float64", "int64"]).columns.to_list()

    # Create a MinMaxScaler object
    mm_scale = MinMaxScaler()

    # Create copies of the train, validate, and test dataframes
    Xtr, Xv, Xt = train.copy(), validate.copy(), test.copy()

    # Add "_mm" to the column names of the columns to be scaled
    scale_mm = [col + "_mm" for col in scale]

    # Scale the columns in the train, validate, and test dataframes
    Xtr[scale_mm] = mm_scale.fit_transform(train[scale])
    Xv[scale_mm] = mm_scale.transform(validate[scale])
    Xt[scale_mm] = mm_scale.transform(test[scale])

    # Drop the unscaled columns from the train, validate, and test dataframes
    Xtr.drop(columns=scale, inplace=True)
    Xv.drop(columns=scale, inplace=True)
    Xt.drop(columns=scale, inplace=True)

    return Xtr, Xv, Xt


def rob_zillow(train, validate, test, scale=None):
    """
    Scale the numerical features of the train, validate, and test datasets using the Robust Scaler method.

    Args:
        train (pandas.DataFrame): training dataset
        validate (pandas.DataFrame): validation dataset
        test (pandas.DataFrame): test dataset
        scale (list): list of columns to scale (default is None, which scales all columns)

    Returns:
        tuple: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled test data)
    """
    # If scale is None, scale all columns
    if scale is None:
        scale = train.columns.to_list()

    # Create a RobustScaler object
    rob_scale = RobustScaler()

    # Create copies of the train, validate, and test dataframes
    Xtr, Xv, Xt = train.copy(), validate.copy(), test.copy()

    # Add "_rob" to the column names of the columns to be scaled
    scale_rob = [col + "_rob" for col in scale]

    # Scale the columns in the train, validate, and test dataframes
    Xtr[scale_rob] = rob_scale.fit_transform(train[scale])
    Xv[scale_rob] = rob_scale.transform(validate[scale])
    Xt[scale_rob] = rob_scale.transform(test[scale])

    # Drop the unscaled columns from the train, validate, and test dataframes
    Xtr.drop(columns=scale, inplace=True)
    Xv.drop(columns=scale, inplace=True)
    Xt.drop(columns=scale, inplace=True)

    return Xtr, Xv, Xt


def metrics_reg(y_true, y_pred):
    """
    Calculate the root mean squared error (RMSE) and R-squared (R2) for a regression model.

    Args:
        y_true (array-like): true target values
        y_pred (array-like): predicted target values

    Returns:
        tuple: RMSE and R2
    """
    # Calculate the RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # Calculate the R2
    r2 = r2_score(y_true, y_pred)

    return rmse, r2


def reg_mods(Xtr, ytr, Xv, yv, features=None, alpha=1, degree=2, power=2):
    """
    Run multiple regression models with different feature combinations, hyperparameters, and output the results as a dataframe.

    Args:
        Xtr (pandas.DataFrame): training features
        ytr (pandas.DataFrame): training target
        Xv (pandas.DataFrame): validation features
        yv (pandas.DataFrame): validation target
        features (list): list of features to use in the models (default: all features)
        alpha (float or list): regularization strength for LassoLars and TweedieRegressor (default: 1)
        degree (int or list): degree of polynomial features for PolynomialFeatures (default: 2)
        power (float or list): power parameter for TweedieRegressor (default: 2)

    Returns:
        pandas.DataFrame: dataframe with the results of each model
    """
    if features is None:
        features = Xtr.columns.to_list()

    # Calculate baseline metrics using mean
    pred_mean = ytr.mean()[0]
    ytr_p = ytr.assign(pred_mean=pred_mean)
    yv_p = yv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(ytr, ytr_p.pred_mean) ** 0.5
    rmse_v = mean_squared_error(yv, yv_p.pred_mean) ** 0.5
    r2_tr = r2_score(ytr, ytr_p.pred_mean)
    r2_v = r2_score(yv, yv_p.pred_mean)
    output = {
        "model": "bl_mean",
        "features": "None",
        "params": "None",
        "rmse_tr": rmse_tr,
        "rmse_v": rmse_v,
        "r2_tr": r2_tr,
        "r2_v": r2_v,
    }
    metrics = [output]

    # Iterate through feature combinations
    for r in range(1, (len(features) + 1)):
        # Cycle through feature combinations for linear regression
        for feature in itertools.combinations(features, r):
            f = list(feature)
            # Linear regression
            lr = LinearRegression()
            lr.fit(Xtr[f], ytr)
            # Metrics
            pred_lr_tr = lr.predict(Xtr[f])
            rmse_tr, r2_tr = metrics_reg(ytr, pred_lr_tr)
            pred_lr_v = lr.predict(Xv[f])
            rmse_v, r2_v = metrics_reg(yv, pred_lr_v)
            # Make it into a table for visualizing later
            output = {
                "model": "LinearRegression",
                "features": f,
                "params": "None",
                "rmse_tr": rmse_tr,
                "rmse_v": rmse_v,
                "r2_tr": r2_tr,
                "r2_v": r2_v,
            }
            metrics.append(output)

        # Cycle through feature combinations and alphas for LassoLars
        for feature, a in itertools.product(itertools.combinations(features, r), alpha):
            f = list(feature)
            # LassoLars
            ll = LassoLars(alpha=a, normalize=False, random_state=123)
            ll.fit(Xtr[f], ytr)
            # Metrics
            pred_ll_tr = ll.predict(Xtr[f])
            rmse_tr, r2_tr = metrics_reg(ytr, pred_ll_tr)
            pred_ll_v = ll.predict(Xv[f])
            rmse_v, r2_v = metrics_reg(yv, pred_ll_v)
            # Make it into a table for visualizing later
            output = {
                "model": "LassoLars",
                "features": f,
                "params": f"alpha={a}",
                "rmse_tr": rmse_tr,
                "rmse_v": rmse_v,
                "r2_tr": r2_tr,
                "r2_v": r2_v,
            }
            metrics.append(output)

        # Cycle through feature combinations and degrees for polynomial feature regression
        for feature, d in itertools.product(
            itertools.combinations(features, r), degree
        ):
            f = list(feature)
            # Polynomial feature regression
            pf = PolynomialFeatures(degree=d)
            Xtr_pf = pf.fit_transform(Xtr[f])
            Xv_pf = pf.transform(Xv[f])
            lp = LinearRegression()
            lp.fit(Xtr_pf, ytr)
            # Metrics
            pred_lp_tr = lp.predict(Xtr_pf)
            rmse_tr, r2_tr = metrics_reg(ytr, pred_lp_tr)
            pred_lp_v = lp.predict(Xv_pf)
            rmse_v, r2_v = metrics_reg(yv, pred_lp_v)
            # Make it into a table for visualizing later
            output = {
                "model": "PolynomialFeature",
                "features": f,
                "params": f"degree={d}",
                "rmse_tr": rmse_tr,
                "rmse_v": rmse_v,
                "r2_tr": r2_tr,
                "r2_v": r2_v,
            }
            metrics.append(output)

        # Cycle through feature combinations, alphas, and powers for TweedieRegressor
        for feature, a, p in itertools.product(
            itertools.combinations(features, r), alpha, power
        ):
            f = list(feature)
            # TweedieRegressor
            lm = TweedieRegressor(power=p, alpha=a)
            lm.fit(Xtr[f], ytr.prop_value)
            # Metrics
            pred_lm_tr = lm.predict(Xtr[f])
            rmse_tr, r2_tr = metrics_reg(ytr, pred_lm_tr)
            pred_lm_v = lm.predict(Xv[f])
            rmse_v, r2_v = metrics_reg(yv, pred_lm_v)
            # Make it into a table for visualizing later
            output = {
                "model": "TweedieRegressor",
                "features": f,
                "params": f"power={p},alpha={a}",
                "rmse_tr": rmse_tr,
                "rmse_v": rmse_v,
                "r2_tr": r2_tr,
                "r2_v": r2_v,
            }
            metrics.append(output)

    return pd.DataFrame(metrics)


def poly_model(X_train, y_train, X_val, y_val, features, degree=3):
    """
    Train a model on the train set and evaluate model using the specified algorithm on the validate set.

    Args:
        X_train (pandas.DataFrame): training features
        y_train (pandas.Series): training target
        X_val (pandas.DataFrame): validation features
        y_val (pandas.Series): validation target
        features (list): list of features to use in the model
        degree (int): degree of polynomial features for PolynomialFeatures (default: 3)
    Returns:
        None
    """
    pf = PolynomialFeatures(degree=degree)
    # Transform the features
    X_train_pf = pf.fit_transform(X_train[features])
    X_val_pf = pf.transform(X_val[features])
    # Create LinearRegression object
    pr = LinearRegression()
    # Fit the model
    pr.fit(X_train_pf, y_train)
    # Make predictions
    pred_pr_tr = pr.predict(X_train_pf)
    pred_pr_v = pr.predict(X_val_pf)
    # Calculate metrics
    rmse_tr, r2_tr = metrics_reg(y_train, pred_pr_tr)
    rmse_v, r2_v = metrics_reg(y_val, pred_pr_v)
    # Print results
    print("Polynomial Features through Linear Regression")
    print(f"Train       RMSE: {rmse_tr}    R2: {r2_tr}")
    print(f"Validate    RMSE: {rmse_v}     R2: {r2_v}")


def poly_test_model(X_train, y_train, X_test, y_test, features, degree):
    """
    Test a polynomial feature regression model using the specified features and targets.

    Args:
        X_train (pandas.DataFrame): training features
        y_train (pandas.Series): training target
        X_test (pandas.DataFrame): test features
        y_test (pandas.Series): test target

    Returns:
        None
    """
    # Select features
    f = features

    # Create PolynomialFeatures object
    pf = PolynomialFeatures(degree=degree)

    # Transform the features
    X_train_pf = pf.fit_transform(X_train[f])
    X_test_pf = pf.transform(X_test[f])

    # Create LinearRegression object
    pr = LinearRegression()

    # Fit the model
    pr.fit(X_train_pf, y_train)

    # Make predictions
    pred_pr_t = pr.predict(X_test_pf)

    # Calculate metrics
    rmse_t, r2_t = metrics_reg(y_test, pred_pr_t)

    # Print results
    print("Polynomial Features through Linear Regression")
    print(f"Test    RMSE: {rmse_t}    R2: {r2_t}")


def plot_predicted_vs_actual(X_train, y_train, X_test, y_test):
    """
    Plot the predicted property values against the actual property values for the test set.

    Args:
        X_train (pandas.DataFrame): training features
        y_train (pandas.Series): training target
        X_test (pandas.DataFrame): test features
        y_test (pandas.Series): test target

    Returns:
        None
    """
    # Select features
    features = [
        "baths_mm",
        "beds_mm",
        "area_mm",
        "age_mm",
        "latitude_mm",
        "longitude_mm",
    ]

    # Create PolynomialFeatures object with degree 3
    poly_features = PolynomialFeatures(degree=3)

    # Transform the features
    X_train = poly_features.fit_transform(X_train[features])
    X_test = poly_features.transform(X_test[features])

    # Create LinearRegression object
    poly_reg = LinearRegression()

    # Fit the model
    poly_reg.fit(X_train, y_train)

    # Make predictions
    y_pred_test = pd.DataFrame(
        poly_reg.predict(X_test), index=y_test.index, columns=["y_pred"]
    )

    # Create a copy of y_test
    pred_mean = y_test.copy()

    # Calculate the mean of prop_value in y_test and assign it to a new column named "baseline"
    pred_mean["baseline"] = y_test["prop_value"].mean()

    # Create a scatter plot of predicted vs. actual property values
    plt.figure(figsize=(16, 8))
    plt.scatter(
        y_test,
        y_pred_test,
        alpha=0.6,
        color="lightblue",
        s=100,
        label="Model 3rd degree Polynomial",
    )

    # Plot the baseline prediction line
    plt.plot(y_test, pred_mean.baseline, color="black", label="_nolegend_")
    plt.annotate("Baseline: Predictions with Mean", (1700000, 200000), fontsize=20)

    # Plot the ideal prediction line
    plt.plot(y_test, y_test, color="blue", label="_nolegend_")
    plt.annotate(
        "The Ideal Line: Predicted = Actual",
        (1800000, 2000000),
        rotation=22,
        fontsize=20,
    )

    # Add labels and title to the plot
    plt.legend()
    plt.xlabel("Actual Property Value (Millions)")
    plt.ylabel("Predicted Property Value (Millions)")
    plt.title("Where are predictions more extreme? More modest?")

    # Show the plot
    plt.show()
