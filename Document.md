**Document:** A document refers to a single piece of text, which can be as small as a sentence or paragraph, or as large as an entire article. In datasets, this often corresponds to one row in a textual dataset.\
Example: A single news article in a collection of news data.

**Corpus:** A corpus is a collection of documents. It represents the entire dataset of text being analyzed or used for training an NLP model.\
Example: A database of 1,000 articles collected for sentiment analysis.

**Vocabulary:** The vocabulary is the set of all unique words present in the corpus. It is used to build features for NLP models and often excludes stop words and rarely occurring terms for efficiency.\
Example: The vocabulary from the phrase “I like apples and oranges” might include {‘I’, ‘like’, ‘apples’, ‘oranges’} if stop words are excluded.

**Tokenization** is the process of splitting text into smaller, manageable units called tokens. These tokens can represent sentences, words, or sub-word units, depending on the level of tokenization.

**Types of Tokenization:**
1.  **Sentence Tokenization:** Splits a text into individual sentences.
    Text: “I love NLP. It’s amazing.”
    Tokens: [“I love NLP.”, “It’s amazing.”]
2.  **Word Tokenization:** Splits a sentence into words or terms.
    Sentence: “I love NLP.”
    Tokens: [“I”, “love”, “NLP”]
3.  **Sub-word Tokenization:** Breaks down words into smaller units like n-grams or Byte Pair Encoding (BPE) tokens. Useful for handling rare or unseen words.
    Word: “unhappiness” → Sub-word Tokens: [“un”, “happi”, “ness”]

**Stop words** are commonly occurring words in a language, such as “is,” “the,” “and,” and “on,” which generally do not provide significant information for NLP tasks.\
Why Remove Stop Words?
*  Reduces noise: These words can overshadow meaningful patterns in text analysis.
*  Reduces dimensionality: Removing them decreases the size of the vocabulary and simplifies computations.
*  Text: “The cat is on the mat.” After removing stop words: “cat mat”

**Stemming** involves chopping off prefixes or suffixes to reduce words to their root form. It’s a rule-based and fast process but may produce non-meaningful root words.\
Words: 
*  “running,” “runner,” “ran” → “run”
*  "studies", "studying" -> studi
*  "universal", "universe", "university" -> univers

**Lemmatization** converts a word to its base or dictionary form (lemma) using vocabulary and grammatical rules. It is more accurate but computationally intensive. Goal to obtain a valid and meaningful base form, crucial for tasks where semantic accuracy is important, such as chatbots and machine translation.\
Words: 
*  “running,” “ran” → “run”
*  "better" -> good
*  "was", "were" -> be

```Key Difference: Stemming is a heuristic process (less accurate), while lemmatization uses linguistic rules (more accurate).```

**POS (Part of Speech) Tagging:** Assigns a grammatical category (noun, verb, adjective, etc.) to each word in a sentence.to reads the text in a language.\
To label terms in text bodies, PoS taggers employ an algorithm. With tags like "noun-plural" or even more complicated labels, these taggers create more complex categories than those stated as basic PoS.

  Sentence: “The quick brown fox jumps over the lazy dog.”\
  Tags: [“The” (Determiner), “quick” (Adjective), “fox” (Noun), “jumps” (Verb), …]\
  Importance: Helps in understanding sentence structure and syntax. Help in downstream NLP tasks like named entity recognition, syntactic parsing, and machine translation. Provides context for ambiguous words. Example: “Book” can be a noun or verb. POS tagging disambiguates it based on context.

**Bag of Words (BOW)** is a simple representation of text where a document is converted into a vector of word frequencies. It disregards grammar, word order, and semantics but focuses solely on the frequency of each word in the document.

How it works:
*   Create a vocabulary of all unique words in the corpus.
*   For each document, count the frequency of each word from the vocabulary.
*   Represent the document as a vector of word counts.

Advantages:
*   Simple and computationally efficient for smaller datasets.
*   Effective for tasks where word occurrence is more important than context (e.g., spam detection).

Limitations:
*   Ignores semantics and word order, meaning it treats “I love NLP” and “NLP love I” as identical.
*   Large vocabularies result in sparse vectors and higher memory usage.

**TF-IDF (Term Frequency-Inverse Document Frequency)** helps us get the importance of a particular word relative to other words in the corpus. It's a common scoring metric in information retrieval (IR) and summarization. TF-IDF converts words into vectors and adds semantic information, resulting in weighted unusual words that may be utilised in a variety of NLP applications.

<<Formula missing>>

**Word embeddings** are dense vector representations of words where similar words have similar vector representations. They capture semantic relationships (e.g., “king” — “man” + “woman” ≈ “queen”).
*  Unlike BOW and TF-IDF, embeddings capture the meaning of words based on their context in a corpus.
*  They are compact (dense vectors) and encode relationships between words.

**Word2Vec** generates embeddings by predicting a word based on its context or vice versa, Focuses on local context windows. Learns word embeddings using neural networks.

Architectures:
1.  **CBOW (Continuous Bag of Words):** Predicts the target word from surrounding context words.\
Example: Given “The __ is barking,” predict “dog.”

2.  **Skip-Gram:** Predicts surrounding words from a target word.\
Example: Given “dog,” predict words like “The,” “is,” and “barking.”

```
*  Word2Vec, GloVe based models build word embedding vectors that are multidimensional.
*  Word2Vec and GloVe are word embeddings, they do not provide any context.
*  Word2Vec and GloVe where existing word embeddings can be used, no transfer learning on
text is possible.
*  GPT is a bidirectional model and word embedding is produced by training on information
flow from left to right.
*  Word2Vec provides simple word embedding i.e. unidirectional language model.
*  Word2Vector algorithm decreases the weight for commonly used words and increases the
weight for words that are not used very much in a collection of documents
```

**GloVe** generates embeddings by factorizing a co-occurrence matrix of word pairs. It learns embeddings based on how often words co-occur in the entire corpus. 
Captures both local (context window) and global (entire corpus) information.
Optimizes word relationships explicitly using co-occurrence statistics.

```
*  BERT (Bidirectional Encoder Representations from Transformer) provides a bidirectional context.
  The BERT model uses the previous and the next sentence to arrive at the context.
*  BERT allows Transform Learning on the existing pre-trained models and hence can be custom trained
for the given specific subject.
*  BERT uses token, segment and position embedding.
*  BERT architecture the relationship between all words in a sentence is modelled irrespective of
their position. --> BERT Transformer architecture models the relationship between each word and all
other words in the sentence to generate attention scores. These attention scores are later used as
weights for a weighted average of all words’ representations which is fed into a fully-connected
network to generate a new representation.
*  
```

```
*  EMLo word embeddings support the same word with multiple embeddings, this helps in using the same
word in a different context and thus captures the context than just the meaning of the word unlike in
GloVe and Word2Vec.Nltk is not a word embedding.
*  ELMo is bidirectional but shallow.
*  ELMo tries to train two independent LSTM language models (left to right and right to left) and
concatenates the results to produce word embedding.
```

```
*  XLNET NLP model has given best accuracy amongst the BERT, GPT-2, ELMo. It has outperformed
BERT on 20 tasks and achieves state of art results on 18 tasks including sentiment analysis,
question answering, natural language inference, etc.
*  XLNET provides permutation-based language modellingand is a key difference from BERT.
In permutation language modeling, tokens are predicted in a random manner and not sequential.
The order of prediction is not necessarily left to right and can be right to left. The original
order of words is not changed but a prediction can be random. 
```

   

# References
1.  https://skphd.medium.com/basic-nlp-interview-questions-and-answers-812289ed2be6
2.  https://www.interviewbit.com/nlp-interview-questions/
3.  https://www.mygreatlearning.com/blog/nlp-interview-questions/
