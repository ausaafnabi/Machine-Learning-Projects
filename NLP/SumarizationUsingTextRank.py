from bs4 import BeautifulSoup
from urllib.request import urlopen

'''
This Function Takes the html input and returns text
@params: url
@return: (title text), textParagraph
'''
def get_only_text(url):
    page=urlopen(url)
    soup=BeautifulSoup(page)
    text=''.join(map(lambda p: p.text,soup.find_all('p')))
    print(text)
    return soup.title,text

def dataInsight(text):
    #Count the number of letters
    print(len(''.join(text)))
    text[:1000] #first 100

from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords
def SummarizerAndkeyworder(text):
    #convert text into string format[explicit]
    text = str(text)
    print('\n\n#Summary:\n\n')
    summary=summarize(text,ratio=0.1)
    print(summary)
    print('\n\n#keywords\n\n')
    print(keywords(text,ratio=0.1))

if __name__ == '__main__':
    print("###################################")
    print("##   Summarizer And Keyworder    ##")
    print("###################################")

    url = input('Enter the Url you want to summarize:\n')
    text = get_only_text(url)
    #dataInsight(text)
    SummarizerAndkeyworder(text)
