# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download necessary NLTK data files (if not already installed)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample document
document = "Running is a good exercise, and it's better than sitting. The quick brown fox jumped over the lazy dog."

# 1. Tokenization
tokens = word_tokenize(document)
print("Tokens:", tokens)

# 2. POS Tagging
pos_tags = pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

# 3. Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
print("\nFiltered Tokens (Stop Words Removed):", filtered_tokens)

# 4. Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Tokens:", stemmed_tokens)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nLemmatized Tokens:", lemmatized_tokens)

# 6. TF and IDF Calculation
# For this step, let's assume we have a corpus of documents. Here, we'll use just one document for simplicity.
corpus = [document]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the document
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Convert the matrix to a DataFrame for easier interpretation
tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=tfidf_vectorizer.get_feature_names_out(), columns=["TF-IDF"])
print("\nTF-IDF Matrix:\n", tfidf_df)