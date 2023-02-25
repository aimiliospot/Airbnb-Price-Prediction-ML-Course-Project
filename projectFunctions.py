import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

def commentSentimentAnalysis():
    reviews = pd.read_csv('./reviews.csv.gz', compression= 'gzip')
    nltk.download('vader_lexicon')
    reviews['comments'] = reviews['comments'].astype('str')
    reviews.comments = reviews.comments.str.replace('<b>', '')
    reviews.comments = reviews.comments.str.replace('</b>', '')
    reviews.comments = reviews.comments.str.replace('<br />', '')
    reviews.comments = reviews.comments.str.replace('<br/>', '')
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    commentsRate = []

    # Analyze sentiment of each review
    for review in reviews['comments']:
        sentiment = sia.polarity_scores(review)
        commentsRate.append(sentiment['compound'])
    
    sentimentPerId = pd.DataFrame()
    sentimentPerId['listing_id'] = reviews['listing_id']
    sentimentPerId['commentRate'] = commentsRate

    grouped = sentimentPerId.groupby('listing_id').mean().reset_index()
    return grouped

def featureSelection(X, numberOfSelectedFeatures):
    selectedFeaturesPerAlgorithm = {}

    XNumerical = X.drop(columns='id')
    for column in XNumerical.columns:
        if XNumerical[column].dtype == 'object' : XNumerical.drop(columns= column, inplace= True)
    XNumerical.dropna(inplace=True)
    XNumerical.reset_index(inplace=True,drop=True)
    price = XNumerical['price']
    XNumerical.drop(columns='price',inplace=True)

    # SelectKBest
    selector = SelectKBest(f_regression, k=numberOfSelectedFeatures)
    selector.fit_transform(XNumerical, price)
    selectedFeaturesPerAlgorithm['SelectKBest'] = list(XNumerical.columns[selector.get_support()])

    # SelectPercentile
    selector = SelectPercentile(mutual_info_regression, percentile = numberOfSelectedFeatures*100/len(XNumerical.columns))
    selector.fit_transform(XNumerical, price)
    selectedFeaturesPerAlgorithm['SelectPercentile'] = list(XNumerical.columns[selector.get_support()])

    # Recursive Feature Elimination
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select = numberOfSelectedFeatures, step= 5)
    selector.fit(XNumerical, price)
    selectedFeaturesPerAlgorithm['RFE'] = list(XNumerical.columns[selector.get_support()])

    print(selectedFeaturesPerAlgorithm)
    return selectedFeaturesPerAlgorithm

def descriptionSentimentAnalysis(descriptions):
   
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    descriptionsRate = []

    # Analyze sentiment of each review
    for description in descriptions:
        sentiment = sia.polarity_scores(description)
        descriptionsRate.append(sentiment['compound'])

    return descriptionsRate

def amenitiesCategorization(X):
    amenities = {
    "kitchen": ["oven", "stove", "kitchen", "kitchenette", "microwave", "grill"],
    "soaps": ["shampoo", "soap", "conditioner", "bathroom essentials",
    "gel", "jel"],
    "fridge": ["refrigerator", "freezer", "fridge"],
    "ac": ["air conditioning", "ceiling fan", "heating", "fans", "heater", "AC"],
    "sound_system": ["sound system"],
    "baby_stuff": ["baby bath", "baby monitor", "baby safety gates", "babysitter recommendations", "crib"],
    "backyard": ["backyard"],
    "waterfront": ["beachfront", "waterfront", "lake access"],
    "bedroom_stuff": ["bed linens", "bedroom comforts", "extra pillows and blankets"],
    "bathroom_stuff": ["bathtub", "bidet"],
    "tv": ["cable tv", "hdtv", "tv"],
    "safety": ["carbon monoxide alarm", "fire extinguisher", "first aid kit", "smoke alarm"],
    "cleaning": ["cleaning products"],
    "kitchen_utilites": ["coffee maker", "hot water kettle", "nespresso machine", "toaster", "bread maker"],
    "workspace": ["workspace", "high chair"],
    "washers": ["dishwasher", "dryer", "washer"],
    "parking": ["carport", "driveway parking", "parking garage", "parking", "garage", "street parking"],
    "gym": ["gym"],
    "pool_and_sauna": ["hot tub", "pool", "sauna"],
    "fireplace": ["fireplace"],
    "patio": ["patio", "balkony"],
    "pets": ["pets"],
    "wifi": ["wifi"]
    }
    amenities_lst = X['amenities']
    for key, values in amenities.items():
        pat = r"|".join(word for word in values)
        amenities_boolean = amenities_lst.str.contains(pat, case=False)
        X.insert(len(X.columns), key, amenities_boolean)
    return X[list(amenities.keys())]