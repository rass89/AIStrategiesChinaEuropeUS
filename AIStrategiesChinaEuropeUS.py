# -*- coding: utf-8 -*-

!pip install PyPDF2
!pip install pyvis
!pip install pyLDAvis

# 1: PDF to text conversion
import PyPDF2

def pdf_to_text(pdf_file, txt_output):
    with open(pdf_file, 'rb') as pdf:
        reader = PyPDF2.PdfReader(pdf)
        with open(txt_output, 'w') as txt:
            for page in range(len(reader.pages)):
                text = reader.pages[page].extract_text()
                txt.write(text)

# Convert PDFs to text files
pdf_to_text('China.pdf', 'China.txt')
pdf_to_text('US.pdf', 'US.txt')
pdf_to_text('EU.pdf', 'EU.txt')

#2: Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Clean the text: remove numbers, URLs, punctuation, and stopwords
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Example usage for each country
with open('China.txt', 'r') as f:
    china_text = f.read()
china_cleaned = clean_text(china_text)

with open('US.txt', 'r') as f:
    us_text = f.read()
us_cleaned = clean_text(us_text)

with open('EU.txt', 'r') as f:
    eu_text = f.read()
eu_cleaned = clean_text(eu_text)

#3: Text analysis
from collections import Counter
from nltk.util import ngrams
import pandas as pd
from scipy.stats import pearsonr

# Function to get the most common n-grams (unigrams or bigrams)
def get_top_ngrams(tokens, n=1, top_n=10):
    n_grams = ngrams(tokens, n)
    return Counter(n_grams).most_common(top_n)

# Get frequency counts for top unigrams and bigrams
china_unigrams = get_top_ngrams(china_cleaned, n=1, top_n=10)
us_unigrams = get_top_ngrams(us_cleaned, n=1, top_n=10)
eu_unigrams = get_top_ngrams(eu_cleaned, n=1, top_n=10)

china_bigrams = get_top_ngrams(china_cleaned, n=2, top_n=10)
us_bigrams = get_top_ngrams(us_cleaned, n=2, top_n=10)
eu_bigrams = get_top_ngrams(eu_cleaned, n=2, top_n=10)

# Convert unigrams/bigrams frequency lists to dictionaries for easy access
def list_to_dict(ngrams_list):
    return {ngram[0]: ngram[1] for ngram in ngrams_list}

china_unigram_dict = list_to_dict(china_unigrams)
us_unigram_dict = list_to_dict(us_unigrams)
eu_unigram_dict = list_to_dict(eu_unigrams)

china_bigram_dict = list_to_dict(china_bigrams)
us_bigram_dict = list_to_dict(us_bigrams)
eu_bigram_dict = list_to_dict(eu_bigrams)

# Merge dictionaries into DataFrame for correlation analysis
def create_freq_df(china_dict, us_dict, eu_dict):
    df = pd.DataFrame([china_dict, us_dict, eu_dict], index=['China', 'US', 'EU']).T.fillna(0)
    return df

# Create frequency DataFrames for unigrams and bigrams
unigram_df = create_freq_df(china_unigram_dict, us_unigram_dict, eu_unigram_dict)
bigram_df = create_freq_df(china_bigram_dict, us_bigram_dict, eu_bigram_dict)

# Print top words and bigrams
print("Top Unigrams in China Document:", china_unigrams)
print("Top Bigrams in China Document:", china_bigrams)

print("Top Unigrams in US Document:", us_unigrams)
print("Top Bigrams in US Document:", us_bigrams)

print("Top Unigrams in EU Document:", eu_unigrams)
print("Top Bigrams in EU Document:", eu_bigrams)

# Calculate Pearson correlation for unigrams and bigrams between countries
def calculate_correlations(df):
    china_us_corr = pearsonr(df['China'], df['US'])[0]
    china_eu_corr = pearsonr(df['China'], df['EU'])[0]
    us_eu_corr = pearsonr(df['US'], df['EU'])[0]

    return china_us_corr, china_eu_corr, us_eu_corr

# Correlations for unigrams
unigram_corr = calculate_correlations(unigram_df)
print(f"Unigram Correlations: China-US: {unigram_corr[0]}, China-EU: {unigram_corr[1]}, US-EU: {unigram_corr[2]}")

# Correlations for bigrams
bigram_corr = calculate_correlations(bigram_df)
print(f"Bigram Correlations: China-US: {bigram_corr[0]}, China-EU: {bigram_corr[1]}, US-EU: {bigram_corr[2]}")

#4. wordcloud generation:
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, country_name):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {country_name}")
    plt.show()

# Example: Generate word clouds
generate_wordcloud(china_cleaned, "China")
generate_wordcloud(us_cleaned, "US")
generate_wordcloud(eu_cleaned, "EU")

import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot correlation heatmap
def plot_correlation(df, title):
    correlation_matrix = df.corr()  # Calculate correlation matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".2f")
    plt.title(title)
    plt.show()

# Plot unigram correlation heatmap
plot_correlation(unigram_df, "Unigram Frequency Correlation Between China, US, and EU")

# Plot bigram correlation heatmap
plot_correlation(bigram_df, "Bigram Frequency Correlation Between China, US, and EU")

# Function to plot stacked bar chart for frequency comparison
def plot_stacked_bar(df, title, ylabel):
    df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
    plt.title(title)
    plt.xlabel("Words/Bigrams")
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot stacked bar chart for unigrams and bigrams
plot_stacked_bar(unigram_df, "Top Unigram Frequency Comparison", "Frequency")
plot_stacked_bar(bigram_df, "Top Bigram Frequency Comparison", "Frequency")

!pip install pyLDAvis==3.3.1 # Install a pyLDAvis version that has the sklearn submodule

import pyLDAvis
# import pyLDAvis.sklearn # This line is removed because it's deprecated in the current pyLDAvis version.
import pyLDAvis.lda_model # Use pyLDAvis.lda_model instead
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# LDA topic modeling
def lda_topic_modeling(texts, n_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    pyLDAvis.enable_notebook()
    # panel = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne') # Changed to use pyLDAvis.lda_model
    panel = pyLDAvis.lda_model.prepare(lda, X, vectorizer, mds='tsne')
    return panel

# Prepare data for LDA
all_texts = [' '.join(china_cleaned), ' '.join(us_cleaned), ' '.join(eu_cleaned)]
lda_panel = lda_topic_modeling(all_texts)

# Save the LDA visualization
pyLDAvis.save_html(lda_panel, 'lda_topic_modeling.html')

#5. Network analysis
import networkx as nx
from pyvis.network import Network

def create_cooccurrence_network(ngrams):
    G = nx.Graph()
    for pair, freq in ngrams:
        G.add_edge(pair[0], pair[1], weight=freq)
    return G

# Visualize the co-occurrence network and save to HTML file
def visualize_network(G, country_name):
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    # Save to an HTML file instead of attempting to render inline
    net.write_html(f"co_occurrence_{country_name}.html")
    print(f"Graph saved as co_occurrence_{country_name}.html")

# Example: Create and visualize networks for China, US, and EU
china_graph = create_cooccurrence_network(china_bigrams)
visualize_network(china_graph, "China")

us_graph = create_cooccurrence_network(us_bigrams)
visualize_network(us_graph, "US")

eu_graph = create_cooccurrence_network(eu_bigrams)
visualize_network(eu_graph, "EU")