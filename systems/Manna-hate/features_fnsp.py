import emoji
import string
import re
from emoji import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count_emoji(text):
    return len([c for c in text if c in UNICODE_EMOJI])

def face_neutral_skeptical(text):
    return len([c for c in text if c in '🤐🤨😐😑😶😏😒🙄😬🤥'])


def face_concerned(text):
    return len([c for c in text if c in '😕😟🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞'])

def fire(text):
    return len([c for c in text if c in '🤑🔥💯💢💥💣✍🗣🌍🌎🌏♨🚨🎯📣📢📷📸📰🗞💰💸💳💵💶💷'])


def face_smiling(text):
    return len([c for c in text if c in '😀😃😄😁😆😅🤣😂🙂🙃😉😊😇'])


def face_affection(text):
    return len([c for c in text if c in '🥰😍🤩😘😗☺😚😙'])


def face_tongue(text):
    return len([c for c in text if c in '😋😛😜🤪😝'])


def face_hand(text):
    return len([c for c in text if c in '🤗🤭🤫🤔'])


def monkey_face(text):
    return len([c for c in text if c in '🙈🙉🙊'])


def emotions(text):
    return len([c for c in text if c in '💋💌💘💝💖💗💓💞💕💟❣💔❤🧡💛💚💙💜🤎🖤'])

def face(text):
    return len([c for c in text if c in '🤐🤨😐😑😶😏😒🙄😬🤥😀😃😄😁😆😅🤣😂🙂🙃😉😊😇😋😛😜🤪😝🤗🤭🤫🤔🙈🙉🙊' ])

def fl(text):
    return len([c for c in text if c in '🇯🇵 🇰🇷 🇩🇪 🇨🇳 🇺🇸 🇫🇷 🇪🇸 🇮🇹 🇷🇺 🇬🇧🇪🇬🇪🇭🇵🇰'])

def preprocess(data):
    print('Preprocessing the Data')

    data['face_concerned'] = data['text'].apply(face_concerned)

    data['flags'] = data['text'].apply(fl)

    data['emotions'] = data['text'].apply(emotions, face_affection)

    data['face_neutral_skeptical'] = data['text'].apply(face_neutral_skeptical)

    data['fire'] = data['text'].apply(fire)

    data['emoji'] = data['text'].apply(count_emoji)

    data['url'] = data['text'].apply(lambda x: len(re.findall('http\S+', x)))

    data['space'] = data['text'].apply(lambda x: len(re.findall(' ', x)))

    data['words'] = data['text'].apply(lambda x: len(re.findall('[a-zA-Z]+', x)))

    data['CapitalLetter'] = data['text'].apply(lambda x: len(re.findall('[A-Z]', x)))

    data['Words_initial_capital'] = data['text'].apply(lambda x: len(re.findall(r"(?<!^)(?<!\. )[A-Z][a-z]+", x)))
    
    data['capital_WORD_count'] = data['text'].apply(lambda x: len(re.findall(r"\b[A-Z][A-Z\d]+\b", x)))

    data['digits'] = data['text'].apply(lambda x: len(re.findall('[0-9]+', x)))

    data['text_length'] = data['text'].apply(len)

    #data['curly_brackets'] = data['text'].apply(lambda x: len(re.findall('[\{\}]', x)))

    data['round_brackets'] = data['text'].apply(lambda x: len(re.findall('[\(\)]', x)))

    #data['quadre_brackets'] = data['text'].apply(lambda x: len(re.findall('\[\]', x)))

    #data['underscore'] = data['text'].apply(lambda x: len(re.findall('[_]', x)))

    data['question_mark'] = data['text'].apply(lambda x: len(re.findall('[?]', x)))

    data['exclamation_mark'] = data['text'].apply(lambda x: len(re.findall('[!]', x)))

    #data['dollar_mark'] = data['text'].apply(lambda x: len(re.findall('[$]', x)))

    #data['currency'] = data['text'].apply(lambda x: len(re.findall('(\d[0-9,.]+)', x)))

    #mon = ['$', 'USD', 'EUR', 'GBP', 'euro', 'euros', 'dollar', 'dollars', 'pound', 'nickel', 'dime', 'pounds', 'money', 'cash']

    #data['amount'] = data['text'].apply(lambda x: len(re.findall("(?=(\b" + '\\b|\\b'.join(mon) + r"\b))", x, flags=re.I)))

    data['ampersand_mark'] = data['text'].apply(lambda x: len(re.findall('&amp;', x))) #&amp;amp;

    data['retweet'] = data['text'].apply(lambda x: len(re.findall('RT', x)))

    data['hashtags'] = data['text'].apply(lambda x: len(re.findall('#HASHTAG#', x)))

    data['url'] = data['text'].apply(lambda x: len(re.findall('#URL#', x)))

    data['first_hash'] = data['text'].apply(lambda x: len(re.findall("^#HASHTAG#", x)))

    data['first_url'] = data['text'].apply(lambda x: len(re.findall("^#URL#", x)))

    data['mentions'] = data['text'].apply(lambda x: len(re.findall('[@]', x)))

    data['usr'] = data['text'].apply(lambda x: len(re.findall('#USER#', x)))

    data['slashes'] = data['text'].apply(lambda x: len(re.findall('[/,\\\\]', x)))

    data['citation'] = data['text'].apply(lambda x: len(re.findall('[\“(.+?)\”]', x))) 

    data['citation2'] = data['text'].apply(lambda x: len(re.findall('[\‘(.+?)\’]', x))) 

    data['ellipsis'] = data['text'].apply(lambda x: len(re.findall('…', x)))

    data['operators'] = data['text'].apply(lambda x: len(re.findall('[+=\-*%<>^|–]', x)))

    data['punct'] = data['text'].apply(lambda x: len(re.findall('[\'\",.:;~`"•]', x)))

    sens = ['watch', 'watch:', 'review', 'says', 'report','reports', 'latest','click', 'follow', 'listen', 'live', 'live now', 'video', 'videos', 'link', 'redirect', 'redirecta', 'direct', 'directa']

    data['adv'] = data['text'].apply(lambda x: len(re.findall("(?=(\b" + '\\b|\\b'.join(sens) + r"\b))", x, flags=re.I)))

    words = ["it", "its", "you", "your", "I", "u", "my", "she", "her", "he", "his", "they", "their", "mine", "myself", "ur", "yo", "tu", "tú", "mi", "toy", "él", "el", "tás", "tas", "tá", "tamos", "tqm", "dtb", "xp", "besit2", "personalmente"]
        
    data['pron_pers'] = data['text'].apply(lambda x: len(re.findall("(?=(\b" + '\\b|\\b'.join(words) + r"\b))", x, flags=re.I)))

    return data


def wordvectorize(data):

    word_vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3)) #1,3 #2,3 #1,4
    word_vectorizer.fit_transform(data['text'])

"""
def charvectorize(data):

    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='char',
    #stop_words='english',
    ngram_range=(1, 3), #2,6 #1,3 #1,4 #1,5
    max_features=50000)
    char_vectorizer.fit_transform(data['text'])
"""
