import re
from string import punctuation 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocess():
    """
    This class is used text preprocessing such as lower cases, removing punctuation, unecessary spaces
    This code is from the BioSyn repository (Sung et al., ACL 2020)
    """

    def __init__(self, lowercase=True, remove_punctuation=True, ignore_punctuations="", stemming=False, typo_path=None):
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc,"")
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(self.punctuation)))

        self.stemming = stemming
        if self.stemming:
            self.stemmer = PorterStemmer() 

    
    def remove_punctuation(self,phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = ' '.join(phrase).strip()

        return phrase

    def stem_tokens(self, text):
        words = word_tokenize(text) 
        
        out = []
        for w in words:
            out.append(self.stemmer.stem(w))
        out = " ".join(out)
        return out

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        # if self.typo_path:
        #     text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)
        
        if self.stemming:
            text = self.stem_tokens(text)

        text = text.strip()

        return text

# def main():
#     preprocessor = TextPreprocess()

#     # text = "adenomatous polyposis coli ( APC ) tumour"

#     # print(preprocessor.run(text))

# if __name__ == '__main__':
#     main()
