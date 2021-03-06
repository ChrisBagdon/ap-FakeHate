import emoji
import string
import re
from emoji import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count_emoji(text):
    return len([c for c in text if c in UNICODE_EMOJI])

def face_neutral_skeptical(text):
    return len([c for c in text if c in '๐ค๐คจ๐๐๐ถ๐๐๐๐ฌ๐คฅ'])


def face_concerned(text):
    return len([c for c in text if c in '๐๐๐โน๐ฎ๐ฏ๐ฒ๐ณ๐ฅบ๐ฆ๐ง๐จ๐ฐ๐ฅ๐ข๐ญ๐ฑ๐๐ฃ๐'])

def fire(text):
    return len([c for c in text if c in '๐ค๐ฅ๐ฏ๐ข๐ฅ๐ฃโ๐ฃ๐๐๐โจ๐จ๐ฏ๐ฃ๐ข๐ท๐ธ๐ฐ๐๐ฐ๐ธ๐ณ๐ต๐ถ๐ท'])


def face_smiling(text):
    return len([c for c in text if c in '๐๐๐๐๐๐๐คฃ๐๐๐๐๐๐'])


def face_affection(text):
    return len([c for c in text if c in '๐ฅฐ๐๐คฉ๐๐โบ๐๐'])


def face_tongue(text):
    return len([c for c in text if c in '๐๐๐๐คช๐'])


def face_hand(text):
    return len([c for c in text if c in '๐ค๐คญ๐คซ๐ค'])


def monkey_face(text):
    return len([c for c in text if c in '๐๐๐'])


def emotions(text):
    return len([c for c in text if c in '๐๐๐๐๐๐๐๐๐๐โฃ๐โค๐งก๐๐๐๐๐ค๐ค'])

def face(text):
    return len([c for c in text if c in '๐ค๐คจ๐๐๐ถ๐๐๐๐ฌ๐คฅ๐๐๐๐๐๐๐คฃ๐๐๐๐๐๐๐๐๐๐คช๐๐ค๐คญ๐คซ๐ค๐๐๐' ])

def fl(text):
    return len([c for c in text if c in '๐ฏ๐ต ๐ฐ๐ท ๐ฉ๐ช ๐จ๐ณ ๐บ๐ธ ๐ซ๐ท ๐ช๐ธ ๐ฎ๐น ๐ท๐บ ๐ฌ๐ง๐ช๐ฌ๐ช๐ญ๐ต๐ฐ'])

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

    data['citation'] = data['text'].apply(lambda x: len(re.findall('[\โ(.+?)\โ]', x))) 

    data['citation2'] = data['text'].apply(lambda x: len(re.findall('[\โ(.+?)\โ]', x))) 

    data['ellipsis'] = data['text'].apply(lambda x: len(re.findall('โฆ', x)))

    data['operators'] = data['text'].apply(lambda x: len(re.findall('[+=\-*%<>^|โ]', x)))

    data['punct'] = data['text'].apply(lambda x: len(re.findall('[\'\",.:;~`"โข]', x)))

    sens = ['watch', 'watch:', 'review', 'says', 'report','reports', 'latest','click', 'follow', 'listen', 'live', 'live now', 'video', 'videos', 'link', 'redirect', 'redirecta', 'direct', 'directa']

    data['adv'] = data['text'].apply(lambda x: len(re.findall("(?=(\b" + '\\b|\\b'.join(sens) + r"\b))", x, flags=re.I)))

    words = ["it", "its", "you", "your", "I", "u", "my", "she", "her", "he", "his", "they", "their", "mine", "myself", "ur", "yo", "tu", "tรบ", "mi", "toy", "รฉl", "el", "tรกs", "tas", "tรก", "tamos", "tqm", "dtb", "xp", "besit2", "personalmente"]
        
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
