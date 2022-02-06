"""
    Description: functions for text preprocessing
"""
import pandas as pd
import re
from bs4 import BeautifulSoup
from string import ascii_letters



# Here is the dictionary that we will use for expanding the contractions:
contraction_mapping =  {"ain't": "is not", 
                        "aren't": "are not",
                        "can't": "cannot", 
                        "'cause": "because", 
                        "could've": "could have", "couldn't": "could not",
                        "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                        "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                        "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                        "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                        "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                        "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                        "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                        "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                        "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                        "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                        "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have"}


def drop_columns(df, columns):
    '''
    Purpose:    drop useless columns in dataframe (df)
    Parameters: the dataframe (df),
                the columns you wanna drop within the dataframe.
                Note: columns should be a list containg the unwanted column names.
    Returns:    the cleaned dataframe (df)
    '''
    for column in columns:
        df = df.drop([column], axis=1)
    return df

# remove tabs ("\n")
def remove_tab(df, column_loc):
    '''
    Purpose:    remove tabs ("\n") in the dataframe's sentences
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = [re.sub(r'\n', ' ', str(text)) for text in df[column_loc]]
    return df

# map contractions
def map_contraction(df, column_loc):
    '''
    Purpose:    map contraction for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = [" ".join([contraction_mapping[t] if t in contraction_mapping else t for t in str(text).split(" ")]) for text in df[column_loc]]
    return df

# remove html tags
def remove_html(df, column_loc):
    '''
    Purpose:    remove html related tags for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = [BeautifulSoup(str(text), "lxml").text for text in df[column_loc]]
    return df


# remove urls and gmails
def remove_urls_and_gmails(df, column_loc):
    '''
    Purpose:    remove url and gmails for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = [re.sub('http://\S+|https://\S+', ' ', str(text)) for text in df[column_loc]]
    # this method cleans the special links:
    # ai-camp.org/ambassador/ashley_forrest
    df[column_loc] = [re.sub(r"(https:/www.)|ai-camp.org/.+?/.[a-z]+|_.[a-z]+", " ", str(text)) for text in df[column_loc]]
    df[column_loc] = [re.sub(r"www.+?.[a-z]+.com", " ", str(text)) for text in df[column_loc]]
    df[column_loc] = [re.sub(r"hello@ai-camp.org", " ", str(text)) for text in df[column_loc]]
    return df

# lower casing
def lower_case(df, column_loc):
    '''
    Purpose:    apply lower casing method for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    # assumes the column has no null
    df[column_loc] = [str(text).lower() for text in df[column_loc]]
    return df

# remove punctuations
allowed = set(ascii_letters + ' ')
def remove_punctuation(df, column_loc):
    '''
    Purpose:    remove punctuation for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = ["".join([c if c in allowed else ' ' for c in str(text)]) for text in df[column_loc]]
    return df


# remove extra spaces
def remove_extra_spaces(df, column_loc):
    '''
    Purpose:    remove extra spaces (" ") for sentences in the dataframe (df)
    Parameters: the dataframe (df),
                the column location of where the sentences are stored (column_loc)
    Returns:    the cleaned dataframe (df)
    '''
    df[column_loc] = [re.sub(r'[" "]+', " ", str(text)) for text in df[column_loc]]
    return df

# function to clean data
def clean_df(df, column_loc, drop_cols=None):
    '''
    Purpose:    apply text preprocessing to the raw csv dataframe
    Parameters: the raw dataframe (df),
                the column location of where the sentences are stored (column_loc),
    Returns:    the cleaned dataframe (df).
    '''
    if drop_cols != None:
        df = drop_columns(df, drop_cols)
    df = df[df[column_loc].isna() == False]                   
    df = remove_tab(df, column_loc)
    df = map_contraction(df, column_loc)
    df = remove_html(df, column_loc)
    df = remove_urls_and_gmails(df, column_loc)
    df = remove_punctuation(df, column_loc)
    df = remove_extra_spaces(df, column_loc)
    df = lower_case(df, column_loc)
    return df


def clean_texts(sents):
    """
    Purpose: take in a list of string sentences needed to be preprocessed before AI model prediction
    Params:  sents, a list of string sentences
    Returns: a list of preprocessed sentences that is preprocessed, but NOT tokenized yet.
    """
    temp_df = pd.DataFrame()
    temp_df['sents'] = sents
    temp_df['sents'] = clean_df(temp_df, 'sents')
    return list(temp_df['sents'].values)


def max_length(texts):
    '''
    Purpose: return the max length each sentence arrays
    Params:  2d array of string.
    Returns: an integer representing the max_length of the sentence arrays
    '''      
    return max(len(t) for t in texts)


def map_labels(label):
    """
    Purpose: a function used for the pandas library's ".apply()" method
             to convert all the specific labels in the dataframe into general labels
    Params:  label(string) -> the label from every single row of the dataframe column
    Returns: a general label (string)
    """
    others = ['ads', 'unique_questions', 'starting_clubs', 'contact_management']
    program_info = ['course_schedule', 'content', 'reschedule']
    registration = ['course_availability', 'application_deadlines', 'payment_confirmation', 'website_navigation', 'account_edits', 'progress_or_spots']
    program_logistics = ['zoom_links', 'zoom_recordings', 'cancel', 'other_urls']
    monetary_issues = ['taxes', 'payment', 'refund']
    scholarship = ['apply_scholarship', 'info_about_scholarship']
    if label in others:
        label = "others"
    elif label in program_info:
        label = "program_info"
    elif label in registration:
        label = "registration"
    elif label in program_logistics:
        label = "program_logistics"
    elif label in monetary_issues:
        label = "monetary_issues"
    elif label in scholarship:
        label = "scholarship"
    elif label == 'unactionable':
        label = 'unactionable'
    return label