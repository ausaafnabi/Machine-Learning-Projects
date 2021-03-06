"""
ProcessRow:
Processes the row (tweet) containing textual data by performing text processing operations.
parameters: string row
returns: string processedRow
"""
def processRow(row):
    import re
    import nltk
    from textblob import TextBlob
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from textblob import Word
    from nltk.util import ngrams
    import re
    from wordcloud import WordCloud, STOPWORDS
    from nltk.tokenize import word_tokenize
    tweet = row
    #Lower case
    tweet.lower()
    #Removes unicode strings like "\u002c" and "x96"
    tweet = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', tweet)
    tweet = re.sub(r'[^\x00-\x7f]',r'',tweet)
    #convert any url to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert any @Username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    #Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]', ' ', tweet)
    #Removes hastag in front of a word """
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove :( or :)
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')
    #remove numbers
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    #remove multiple exclamation
    tweet = re.sub(r"(\!)\1+", ' ', tweet)
    #remove multiple question marks
    tweet = re.sub(r"(\?)\1+", ' ', tweet)
    #remove multistop
    tweet = re.sub(r"(\.)\1+", ' ', tweet)
    #lemma
    from textblob import Word
    tweet =" ".join([Word(word).lemmatize() for word in tweet.split()])
    #stemmer
    #st = PorterStemmer()
    #tweet=" ".join([st.stem(word) for word in tweet.split()])
    #Removes emoticons from text
    '''tweet = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X
            -p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\
            (|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-
            \*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\
            (|>\.<|>=\(|D:',"", tweet)
   '''#trim
    tweet = tweet.strip('\'"')
    processedRow = tweet
    return processedRow
    
# Example
## tweet= "How to take control of your #debt https://personal.vanguard.com/us/insights/saving-investing/debt-­management.#Best advice for #family #financial #success ­(@PrepareToWin)"
## processRow(tweet)
## Output:
## How to take control of your debt URL advice for family financial success AT_USER
# Time Taken for the script: 3s
# improvement regarding efficiency

