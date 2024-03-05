"""
Description:
    Preprocessing functions to run the models

Date:
    30th September, 2022
"""
import sys
import csv
import pandas as pd
import numpy as np
from numpy import array
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from maps import mapDict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from decimal import Decimal

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import pydotplus

def get_classification_samples(test_x, testy, prediction, cols, model_name):
    """
    Args:
        - Get true positive, true negative, false positive, false negative samples 
        - Save with model name and nature of the samples
    Return:
        - Dataframes
    """
    true_positive_indices = (testy == 1) & (prediction == 1)
    true_negative_indices = (testy == 0) & (prediction == 0)
    false_negative_indices = (testy == 1) & (prediction == 0)
    false_positive_indices = (testy == 0) & (prediction == 1)

    tp_df = pd.DataFrame(test_x[true_positive_indices], 
                    columns=cols)

    tn_df = pd.DataFrame(test_x[true_negative_indices], 
                    columns=cols)

    fn_df = pd.DataFrame(test_x[false_negative_indices], 
                    columns=cols)

    fp_df = pd.DataFrame(test_x[false_positive_indices], 
                    columns=cols)

    return tp_df, tn_df, fn_df, fp_df


def create_labels(_df, label):
    """
    Description:
        Create binary categories from
        multiple categories.
    Args:
        df (dataframe)
        label (string): primary label
    Returns:
        df (dataframe): a dataframe with additional
        attributes
    """
    label2 = 'No intervention'
    positive_class = _df[_df['cluster'].isin([label])]
    negative_class = _df[_df['cluster'].isin([label2])]

    positive_class['label'] = ['positive']*len(positive_class)
    negative_class['label'] = ['negative']*len(negative_class)
    _df = pd.concat([positive_class, negative_class])
    return _df

def is_zero(value):
    if value == 0:
        return True
    else:
        return False

def is_negative(value):
    if value < 0:
        return True
    else:
        return False
def is_one(value):
    if value == 1:
        return True
    else:
        return False

def is_low(value):
    if value > 0 and value < 0.125:
        return True
    else:
        return False

def is_medium(value):
    if value > 0.125 and value < 0.33:
        return True
    else:
        return False

def is_high(value):
    if value >= 0.33:
        return True
    else:
        return False

def provide_label(sub, deck, sup):
    """
    Description:
        Return the label for the value of
    subsructure, superstructure, and deck
    """
    componentDict = {0:'Substructure',
                     1:'Deck',
                     2:'Superstructure'}

    values = [sub, deck, sup]
    labels = list()
    for num in range(len(values)):
        value = float(values[num])
        if is_zero(value):
            label = 'No ' + componentDict[num]
        elif is_low(value):
            label = 'Yes ' + componentDict[num]
        elif is_negative(value):
            label = 'Yes ' + componentDict[num]
        elif is_medium(value):
            label = 'No ' + componentDict[num]
        elif is_high(value):
            label = 'Yes ' + componentDict[num]
        else:
            label = 'Error ' + componentDict[num]
        labels.append(label)
    return labels


def semantic_labeling_utility(record):
    """
    Description:
       Utility to assign a label depending on the values
    """
    sub, deck, sup = record
    if sub == 0 and deck == 0 and sup == 0:
        label = "No intervention"
    elif sub == 1 and deck == 1 and sup == 1:
        label = 'All intervention'
    elif sub > 0 and deck > 0 and sup > 0:
        label = provide_label(sub, deck, sup)
    else:
        label = "Other intervention"
        label = provide_label(sub, deck, sup)
    return label

def semantic_labeling(features, name=""):
    """
    Description:
       Assign a semantic label
    """
    labels = list()
    for index, record in features.iterrows():
        subInt = record['subNumberIntervention']
        deckInt = record['deckNumberIntervention']
        supInt = record['supNumberIntervention']

        label = semantic_labeling_utility([subInt,
                                          deckInt,
                                          supInt])
        if type(label) is type(list()):
            label = ' - '.join(label)
        labels.append(label)
    return labels

def geo_coor_utility(longitude, latitude):
    """
    Function for converting longitude and latitude
    args:
        longitude: NBI format
        latitude: NBI format

    return:
        long, lat: return converted longitude and latitude
    """
    if latitude > 0 and longitude > 0:
        lat = str(latitude)
        lat = lat.rjust(8, '0')
        lat_degree = Decimal(lat[:2])
        lat_min = Decimal(lat[2:4])
        lat_min = (lat_min/60)
        lat_sec = Decimal(lat[4:8])
        lat_sec = (lat_sec/360000)
        lat_decimal = lat_degree + lat_min + lat_sec

        long = str(longitude)
        long = long.rjust(9, '0')
        if len(long) <= 9:
            long  = long.zfill(9)
        long_degree = Decimal(long[:3])
        long_min = Decimal(long[3:5])
        long_min = (long_min/60)
        long_sec = Decimal(long[5:9])
        long_sec = (long_sec/360000)
        long_decimal = - (long_degree + long_min + long_sec)
        long_decimal =  format(long_decimal, '.6f')
        lat_decimal =  format(lat_decimal, '.6f')
        return long_decimal, lat_decimal
    return 0.00, 0.00

def convert_geo_coordinates(_df, columns):
    """
    Function for converting longitude and latitude
    args:
        df: Dataframe
        columns: all the column

    return:
        df: dataframe with decimal longitude and latitude
    """
    longitudes = _df['longitude']
    latitudes = _df['latitude']

    trans_longitudes = []
    trans_latitudes =[]

    for longitude, latitude in zip(longitudes, latitudes):
        t_longitude, t_latitude = geo_coor_utility(longitude,
                                                latitude)
        trans_longitudes.append(t_longitude)
        trans_latitudes.append(t_latitude)

    _df['longitude'] = trans_longitudes
    _df['latitude'] = trans_latitudes
    return _df

def one_hot(_df, columns):
    """
    Function for one-hot-encoding
    args:
        df: Dataframe
        columns: all the columns
        fields: specific fields to code one hot encoding

    return:
        columns: columns with one hot encoding
    """
    map_dict = mapDict
    for column in columns:
        col_map = map_dict[column]
        data = _df[column].map(col_map)
        values = array(data)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        # binary encoder
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # define a new dictionary
        dict_col = defaultdict(list)
        for row in onehot_encoded:
            for index, value in zip(label_encoder.classes_, row):
                index = column + str(index)
                dict_col[index].append(value)
        for key in dict_col.keys():
            _df[key] = dict_col[key]

        #TODO: Next, we have to figure out how do we scale these to other material
        # One-hot encoding categorial variable with high cardinality
        # Cause inefficiency in tree-based ensembles.
        # Continuous variables will be given more importance 
        # than the dummy variables by the algorithm
        # which will obscure the order of feature
        # importance resulting in poorer performance.
        # Further, we need to look at feature hasher and how it can help
        # Categorical variables: Material
    return _df

def read_csv(csv_file):
    """
    Read csv files using pandas
    """
    _df = pd.read_csv(csv_file)
    return _df

def create_label(intervention_columns):
    """
    Return positive or negative function
    """
    labels = []
    for value in intervention_columns:
        if value == 0:
            label_val = 'negative'
            labels.append(label_val)
        else:
            label_val = 'positive'
            labels.append(label_val)
    return labels

def normalize(_df, columns):
    """
    Function for normalizing the data

    Args:
        _df (dataframe)
        columns (features)
    """
    for feature in columns:
        _df[feature] = _df[feature].astype(int)
        max_value = _df[feature].max()
        min_value = _df[feature].min()
        _df[feature] = (_df[feature] - min_value) / (max_value - min_value)
    return _df

def remove_null_values(_df):
    """
    Description: return a new df with null values removed
    Args:
        df (dataframe): the original dataframe to work with
    Returns:
        df (dataframe): dataframe
    """
    for feature in _df:
        if feature != 'structureNumber':
            try:
                _df = _df[~_df[feature].isin([np.nan])]
            except:
                print("Error: ", feature)
    return _df


def remove_duplicates(_df, column_name='structureNumbers'):
    """
    Description: return a new df with duplicates removed
    Args:
        df (dataframe): the original dataframe to work with
        column (string): columname to drop duplicates by
    Returns:
        newdf (dataframe)
    """
    temp = []
    for group in _df.groupby(['structureNumber']):
        structure_number, grouped_df = group
        grouped_df = grouped_df.drop_duplicates(subset=['structureNumber'],
                               keep='last'
                               )
        temp.append(grouped_df)
    new_df = pd.concat(temp)
    return new_df

def preprocess(csv_file = '../data/nebraska_deep.csv'):
    # Read CSV
    df = read_csv(csv_file)

    print("printing shape of df:", len(df))
    # Remove null values:
    df = df.dropna(subset=['deck',
                           'substructure',
                           'deckNumberIntervention',
                           'subNumberIntervention',
                           'supNumberIntervention'
                          ])

    print("Before removing duplicates printing shape of df:", len(df))
    df = remove_duplicates(df)
    print("printing shape of df:", len(df))

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    df = df[~df['deckStructureType'].isin(['N', 'U'])]

    print("After removing 'N' and 'U' values printing shape of df:", len(df))
    # Fill the null values with -1:
    #df.snowfall.fillna(value=-1, inplace=True)
    #df.snowfall
    #df.precipitation.fillna(value=-1, inplace=True)
    #df.freezethaw.fillna(value=-1, inplace=True)

    #df.toll.fillna(value=-1, inplace=True)
    df.designatedInspectionFrequency.fillna(value=-1, inplace=True)
    df.deckStructureType.fillna(value=-1, inplace=True)
    #df.typeOfDesign.fillna(value=-1, inplace=True)

    # Normalize features:
    columns_normalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention",
                        "latitude",
                        "longitude",
                        "skew",
                        "numberOfSpansInMainUnit",
                        "lengthOfMaximumSpan",
                        "structureLength",
                        "bridgeRoadwayWithCurbToCurb",
                        "operatingRating",
                        "scourCriticalBridges",
                        "lanesOnStructure",
                        ]

    # Select final columns:
    columns_final = [
                    "structureNumber",
                    "yearBuilt",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                    "material",
                    "designLoad",
                    "snowfall",
                    "freezethaw",
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention",

                    "latitude",
                    "longitude",
                    "skew",
                    "numberOfSpansInMainUnit",
                    "lengthOfMaximumSpan",
                    "structureLength",
                    "bridgeRoadwayWithCurbToCurb",
                    "operatingRating",
                    "scourCriticalBridges",
                    "lanesOnStructure",

                    "toll",
                    "designatedInspectionFrequency",
                    "deckStructureType",
                    "typeOfDesign",
                ]

    cols = columns_normalize
    #data_scaled = normalize(df, columns_normalize)
    print("[Before scaling] printing shape of df:", len(df))

    data_scaled = df[columns_final]
    print("[After scaling] printing shape of df:", len(df))

    temp_columns_hot_encoded = {'material': 'CatMaterial',
                                "toll": 'CatToll',
                                "designLoad": 'CatDesignLoad' ,
                                "deckStructureType": 'CatDeckStructureType',
                                "typeOfDesign":'CatTypeOfDesign'
                              }

    data_scaled.rename(columns=temp_columns_hot_encoded, inplace=True)
    columns_hot_encoded = temp_columns_hot_encoded.values()
    data_scaled = one_hot(data_scaled, columns_hot_encoded)

    print("[Before removing null values] printing shape of df:", len(data_scaled))
    data_scaled = remove_null_values(data_scaled)

    print("[After removing null values] printing shape of df:", len(data_scaled))
    data_scaled = convert_geo_coordinates(data_scaled, ['longitude', 'latitude'])

    columns_final = list(data_scaled.columns)
    columns_final.remove('CatMaterial')
    columns_final.remove('CatToll')
    columns_final.remove('CatDesignLoad')
    columns_final.remove('CatDeckStructureType')
    columns_final.remove('CatTypeOfDesign')

    #TODO: Apply recursive feature elimination here.
    # Data Scaled
    features = ["structureNumber",
                "supNumberIntervention",
                "subNumberIntervention",
                "deckNumberIntervention"]

    s_labels = semantic_labeling(data_scaled[features],
                                name="")
    print("Categorical labels")
    print(Counter(s_labels))
    columns_final.remove('structureNumber')
    features.remove('structureNumber')
    structure_number = data_scaled['structureNumber']
    data_scaled = data_scaled[columns_final]
    data_scaled['cluster'] = s_labels

    # Deck Number Intervention -> 0, 1.
    # TODO: 
        # 1. One hot encoding
        # 2. Creating consistent label using deck, superstructure and substructure

    #deck_labels = X['deckNumberIntervention']
    columns_final.remove('deckNumberIntervention')
    columns_final.remove('supNumberIntervention')
    columns_final.remove('subNumberIntervention')


    labels = ['No Substructure - Yes Deck - No Superstructure',
              'Yes Substructure - No Deck - No Superstructure',
              'No Substructure - No Deck - Yes Superstructure']

    #label = 'No Substructure - Yes Deck - No Superstructure'
    #label = 'Yes Substructure - No Deck - No Superstructure'
    label = 'No Substructure - No Deck - Yes Superstructure'

    print("[Before creating labels] length of data scaled", len(data_scaled))
    data_scaled = create_labels(data_scaled, label)
    print("[After creating labels] length of data scaled", len(data_scaled))
    clusters = Counter(data_scaled['label'])

    list_of_clusters = list()
    for cluster in clusters.keys():
        num_of_members = clusters[cluster]
        if num_of_members < 15:
            list_of_clusters.append(cluster)
    data_scaled = data_scaled[~data_scaled['label'].isin(list_of_clusters)]

    print("[Before X, Y split] length of data scaled", len(data_scaled))
    X, y = data_scaled[columns_final], data_scaled['label']
    neg = data_scaled[data_scaled['label'] == 'negative']
    pos = data_scaled[data_scaled['label'] == 'positive']

    # List of categorical columns
    categorical_col = [
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                            28,
                            29,
                            30,
                            31,
                            32,
                            33,
                            34,
                            35,
                            36,
                            37,
                            38,
                            39,
                            40,
                            41,
                            42,
                            43,
                            44,
                            45,
                            46,
                            47,
                            48
        ]

    # Sampling Techniques
    #sampling = SMOTE()
    #sampling = SMOTEN(random_state=0)
    #sampling = SMOTENC(random_state=42,
    #                  categorical_features=categorical_col)
    #sampling = RandomUnderSampler(sampling_strategy='auto')

    #X, y = sampling.fit_resample(X, y)

    # Convert them into arrays
    X = np.array(X)
    y = np.array(y)

    holdouts = []

    print('length of X', len(X))
    return X, y, holdouts, columns_final

