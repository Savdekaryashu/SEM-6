# Import necessary libraries
import nltk
import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords,wordnet


# # Download required NLTK data files (only need to run these once)
# nltk.download('punkt_tab')       # Tokenizer models
# nltk.download('wordnet')      # WordNet corpus for lemmatization
# nltk.download('stopwords')    # Stopwords list
nltk.download('omw-1.4')
# Sample text to preprocess
text = "This is an example sentence, to demonstrate preprocessing in NLP. Let's test the regex-based tokenization! intelligent"

# 1. Tokenization using NLTK's word_tokenize
#    This function splits the text into individual words and punctuation.
tokens = word_tokenize(text)
print("Word Tokenization using NLTK:", tokens)

# 2. Tokenization using Regular Expressions
#    Here we use RegexpTokenizer to extract words (ignoring punctuation).
tokenizer = RegexpTokenizer(r'\w+')
regex_tokens = tokenizer.tokenize(text)
print("Tokenization using Regular Expressions:", regex_tokens)

# 3. Stop word removal
#    Stop words (common words like 'is', 'and', etc.) are filtered out.
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in regex_tokens if token.lower() not in stop_words]
print("Tokens after Stopword Removal:", filtered_tokens)

# 4. Stemming
#    Using PorterStemmer to reduce words to their base or root form.
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# 5. Lemmatization
#    Using WordNetLemmatizer to obtain the dictionary form (lemma) of each word.
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
