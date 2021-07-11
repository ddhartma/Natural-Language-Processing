[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 
[image19]: assets/19.png 
[image20]: assets/20.png 
[image21]: assets/21.png 
[image22]: assets/22.png 
[image23]: assets/23.png 
[image24]: assets/24.png 
[image25]: assets/25.png 
[image26]: assets/26.png 
[image27]: assets/27.png 
[image28]: assets/28.png 
[image29]: assets/29.png 
[image30]: assets/31.png 


# Natural Language Processing (NLP)

Overview of Natural Language Processing techniques.

Please check my [Data Science - NLP](https://github.com/ddhartma/NLP-Pipelines) repository  

## Content 
- [Preprocessing for NLP](#preprocess)
    - [Cleaning](#cleaning)
    - [Tokenization](#tokenization)    
    - [Normalization](#normalization)  
    - [Removing Interpunctuations](#rem_interpunct)  
    - [Stop Words Removal](#stop_wors_removal)
    - [Stemming](#stem)
    - [N-Grams](#n_gram)
    - [Example for a combined NLP preprocessing](#exp_preprocess)
- [Word Embeddings with word2vec](#word_2_vec)
    - [Principal idea of word2vec](#idea_word2vec)
    - [Execute word2vec](#execute_word2vec)
    - [Plot word vectors](#plot_word_vecs)
- [Receiver Operating Characteristic - ROC](#roc)
    - [Truth table - Confusion matrix](#truth_table)
    - [Calculating ROC-AUC-Metric](#roc_calc)
- [Natural Language Classification with Familiar Networks](#nlp_class_a)
    - [Loading the IMDB Film Reviews](#loading_imdb)
    - [Examining the IMDB Data](#examine_imdb)
    - [Standardizing the Length of the Reviews](#standardize_imdb)
    - [Dense Network](#dense_net)
    - [Convolutional Networks](#conv_net)
- [Networks Designed for Sequential Data](#network_seq)
    - [Recurrent Neural Networks (RNNs)](#rnn)
    - [Long Short-Term Memory Units (LSTMs)](#lstm) 
    - [Bidirectional LSTMs (Bi-LSTMs)](#bidirec_lstm)
    - [Stacked Recurrent Models](#stacked_rnn) 
    - [seq2seq and Attention](#seq2seq_attentiom)
    - [Transfer Learning in NLP](#transfer_learn_nlp)
-  [Non-Sequential Architectures: The Keras Functional API](#non_seq_architect)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Preprocessing for NLP <a id="preprocess"></a>
## Cleaning <a id="cleaning"></a>  
- Open notebook ```cleaning.ipynb``` to handle text cleaning
- Let's walk through an example of cleaning text data from a popular source - the web. There are helpful tools in working with this data, including the
  - [requests library](https://2.python-requests.org/en/master/user/quickstart/#make-a-request)
  - [regular expressions](https://docs.python.org/3/library/re.html)
  - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

- Request a web page
  ```
  import requests
  # fetch web page
  r = requests.get('https://www.udacity.com/courses/all')
  ```

  Downloaded successfully (status = 200)
  ```
  r.status_code
  ```
- Parsing a Page with Beautifulsoup
  ```
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(r.text, "lxml")
  soup.text
  ```

  Print with readable indent
  ```
  print(soup.prettify())
  ```

  List all tags that are nested
  ```
  list(soup.children)
  ```

  List children of children
  ```
  html = list(soup.children)[2]
  body = list(html.children)[3]
  p = list(body.children)[1]
  ```

  Get text from children
  ```
  p.get_text()
  ```
- Finding all instances of a tag at once
  ```
  soup.find_all('p')
  ```

  Use list indexing, it to extract text:
  ```
  soup.find_all('p')[0].get_text()
  ```

  If only first p instance is needed
  ```
  soup.find('p')
  ```

- Searching for tags by class and id
  Search for any ```p``` tag that has the ```class='outer-text'```
  ```
  soup.find_all('p', class_='outer-text')
  ```

  Search for any tag that has the ```class='outer-text'```
  ```
  soup.find_all(class_="outer-text")
  ```

  Find id
  ```
  soup.find_all(id="first")
  ```

  CSS selectors to find all the p tags in a page that are inside of a div
  ```
  soup.select("div p")
  ```
## Tokenization <a id="tokenization"></a> 
- Open notebook ```tokenization.ipynb``` to handle text tokenization
- Tokenization is simply splitting text into sentences or a sentence into a sequence of words.
- Simple method: ```split()```
  ```
  # Split text into words
  words = text.split()
  print(words)
  ```
- [NLTK](http://www.nltk.org/book/) library - NATURAL LANGUAGE TOOLKIT

  - It is smarter e.g. in terms of punctuation ['Dr.', 'Smith', 'graduated', ... , '.']
  - Split text into words
    ```
    from nltk.tokenize import word_tokenize
    # split text into words using NLTK
    words = word_tokenize(text)
    ```
  - Split text into sentences (e.g. for translation)
    ```
    from nltk.tokenize import sent_tokenize
    # split text into sentences using NLTK
    sentences = sent_tokenize(text)
    ```

  - NLTK has several other options e.g.
    - a regular expression base tokenizer to ***remove punctuation*** and ***perform tokenization*** in a single step.
    - a tweet tokenizer that is aware of twitter handles, hash tags and emoticons   
## Normalization <a id="normalization"></a>  
- Open notebook ```normalization.ipynb``` to handle text normalization
- ***Case Normalization***: In Machine Learning it does not make sense to differentiate between 'car', 'Car' and 'CAR'. These all three words have the same meaning. Therefore: Normalize all words to lower case
  ```
  text = text.lower()
  ```
- Pay attention: For example in German language lower case and upper case of words change the meaning ("general" like "allgemein" or "General"). Normalization is not always good.

## Removing Interpunctuations <a id="rem_interpunct"></a> 
- Open notebook ```normalization.ipynb``` to handle text normalization
- ***Punctual Removal***: Dependenfing on the NLP task, one wants to remove special characters like periods, question marks, exclamation points and only keep letters of the alphabet and maybe numbers (especially usefull for document classification and clustering where low level details do not matter a lot)
  ```
  import re
  # Remove punctuation from text and
  # only keep letters of the alphabet and maybe numbers
  # everything else is replaced by a space

  text = re.sub(r"[^a-zA-Z0-9]", " ", text)
  ```

- Pay attention: For example, if you want to train an algorithm to answer questions, question marks are important to identify questions.

## Stop Words Removal <a id="stop_wors_removal"></a> 
- Open notebook ```stop_words.ipynb``` to handle stop word removal
- Stop words are uninformative words like ***is. our, the, in, at, ...*** that do not add a lot of meaning to a sentence.
- Remove them to reduce the vocabulary size (complexity of later procedures)
- [NLTK](http://www.nltk.org/book/) library  can identify stop words
  ```
  # List stop words
  from nltk.corpus import stopwords
  print(stopwords.words("english"))
  ```
- Remove stop words with a Python list comprehension with a filtering condition
  ```
  # Remove stop words
  words = [w for w in words if w not in stopwords.words("english")]
  ```
- Pay attention: It is not always a good idea to remove stop words. For example, in sentiment anlysis stop words could give a hint on the sentiment.

## Stemming and Lemmatization <a id="stem"></a> 
- Open notebook ```stem_lem.ipynb``` to handle Stemming and Lemmatization.
- ***Stemming***: In order to further simplify text data, stemming is the process of reducing a word to its stem or root form.
- For instance, branching, branched, branches et cetera, can all be reduced to branch.
- the suffixes 'ing' and 'ed' can be dropped off, 'ies' can be replaced by 'y' et cetera.
- Stemming is meant to be a fast operation.
- NLTK has a few different stemmers for you to choose from:
    - PorterStemmer
    - SnowballStemmer
    - other language-specific stemmers


- PorterStemmer (remove stop words beforehand)
  ```
  from nltk.stem.porter import PorterStemmer

  # Reduce words to their stem
  stemmed = [PorterStemmer().stem(w) for w in words]
  print(stemmed)
  ```
- ***Lemmatization***: This is another technique to reduce words to a normalized form.
- In this case the transformation uses a ***dictionary*** to map different variants of a word back to its root.
- With this approach, we are able to reduce non-trivial inflections such as 'is', 'was', 'were', back to the root 'be'.
- [NLTK](http://www.nltk.org/book/) uses the default lemmatizer Wordnet database.
  ```
  from nltk.stem.wordnet import WordNetLemmatizer

  # Reduce words to their root form
  lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
  print(lemmed)
  ```

-  A lemmatizer needs to know about the part of speech for each word it's trying to transform. In this case, WordNetLemmatizer defaults to nouns, but one can override that by specifying the **pos** parameter. Let's pass in 'v' for verbs.
  ```
  Lemmatize verbs by specifying pos
  lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
  print(lemmed)
  ```

- Stemming sometimes results in stems that are not complete words in English. Lemmatization is similar to stemming with one difference, the final form is also a meaningful word. Stemming does not need a dictionary like lemmatization does. Stemming maybe a less memory intensive option.

  ![image1]

- Pay attention: Sometimes stemming or lemmatization may not be applied to the text. For example, if the text is sufficiently large.

### Summary of Text Processing <a name="Summary_of_Text_Processing"></a>
 1. Normalize
 2. Tokenize
 3. Remove Stop Words
 4. Stem / Lemmatize

    ![image2]

## N-Grams <a id="n_gram"></a> 
- Open Jupyter Notebook ```natural_language_preprocessing```
- N-Grams: Some words exist in combination with others
- Examples:
    - Bi-gram: **New York** 
    - Tri-gram: **New York City**

    ### Load necessary libraries
    ```
    import nltk
    from nltk import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import *
    nltk.download('gutenberg')
    nltk.download('punkt')
    nltk.download('stopwords')

    import string

    import gensim
    from gensim.models.phrases import Phraser, Phrases
    from gensim.models.word2vec import Word2Vec
    ``` 
    ### Tokenization step 
    ```
    # a convenient method that handles newlines, as well as tokenizing sentences and words in one shot
    gberg_sents = gutenberg.sents()
    ```
    ### Handle bigram collocations
    ```
    # Train a detector: how often occur word pairs in relation to single occurences of both words.
    phrases = Phrases(gberg_sents) 
    
    # Create a more efficient Phraser object for transforming sentences: combine two related tokens to one token
    bigram = Phraser(phrases) 
    
    # Output count and score of each bigram
    bigram.phrasegrams 

    RESULTS: Miss Taylor with a high score of 454 --> Real Bigram
    ------------
    {(b'two', b'daughters'): (19, 11.966813731181546),
    (b'her', b'sister'): (195, 17.7960829227865),
    (b"'", b's'): (9781, 31.066242737744524),
    (b'very', b'early'): (24, 11.01214147275924),
    (b'Her', b'mother'): (14, 13.529425062715127),
    (b'long', b'ago'): (38, 63.22343628984788),
    (b'more', b'than'): (541, 29.023584433996874),
    (b'had', b'been'): (1256, 22.306024648925288),
    (b'an', b'excellent'): (54, 39.063874851750626),
    (b'Miss', b'Taylor'): (48, 453.75918026073305),
    (b'very', b'fond'): (28, 24.134280468850747),
    (b'passed', b'away'): (25, 12.35053642325912),
    (b'too', b'much'): (173, 31.376002029426687),
    (b'did', b'not'): (935, 11.728416217142811),
    (b'any', b'means'): (27, 14.096964108090186),
    (b'wedding', b'-'): (15, 17.4695197740113),
    (b'Her', b'father'): (18, 13.129571562488772),
    (b'after', b'dinner'): (21, 21.5285481168817),
    ...
    }
    ```
    ### Tokenize + Bigram a new sentence
    ```
    # Tokenize via split method
    tokenized_sentence = "Jon lives in New York City".split()
    print(tokenized_sentence)

    # Search for bigrams and transform them to single tokens
    bigram[tokenized_sentence]

    RESULTS:
    ------------
    ['Jon', 'lives', 'in', 'New', 'York', 'City']
    ['Jon', 'lives', 'in', 'New_York', 'City']
    ```

## Example for a combined NLP preprocessing <a id="exp_preprocess"></a> 
- Open Jupyter Notebook ```natural_language_preprocessing```
- Apply Normalization and removal of interpunctuation 
    ```
    # - leave in stop words ("indicative of sentiment")
    # - no stemming ("model learns similar representations of words of the same stem when data suggests it")
    # - apply normalization
    # - apply removal of interpunctuation 
    lower_sents = []
    for s in gberg_sents:
        lower_sents.append([w.lower() for w in s if w.lower()
                            not in list(string.punctuation)])

    # Search for bigrams
    lower_bigram = Phraser(Phrases(lower_sents, 
                               min_count=32, threshold=64))
    lower_bigram.phrasegrams

    RESULTS:
    ------------
    {(b'miss', b'taylor'): (48, 156.44059469941823),
    (b'mr', b'woodhouse'): (132, 82.04651843976633),
    (b'mr', b'weston'): (162, 75.87438262077481),
    (b'mrs', b'weston'): (249, 160.68485093258923),
    (b'great', b'deal'): (182, 93.36368125424357),
    (b'mr', b'knightley'): (277, 161.74131790625913),
    (b'miss', b'woodhouse'): (173, 229.03802722366902),
    (b'years', b'ago'): (56, 74.31594785893046),
    (b'mr', b'elton'): (214, 121.3990121932397),
    (b'dare', b'say'): (115, 89.94000515807346),
    (b'frank', b'churchill'): (151, 1316.4456593286038),
    (b'miss', b'bates'): (113, 276.39588291692513),
    (b'drawing', b'room'): (49, 84.91494947493561),
    (b'mrs', b'goddard'): (58, 143.57843432545658),
    (b'miss', b'smith'): (58, 73.03442128232508),
    (b'few', b'minutes'): (86, 204.16834974753786),
    (b'john', b'knightley'): (58, 83.03755747111268),
    ...
    }
    ```
    ### Create corpus with cleaned sentences
    ```
    clean_sents = []
    for s in lower_sents:
        clean_sents.append(lower_bigram[s])

    print(lean_sents[6])

    RESULTS:
    ------------
    ['sixteen',
    'years',
    'had',
    'miss_taylor',
    'been',
    'in',
    'mr_woodhouse',
    's',
    'family',
    'less',
    'as',
    'a',
    'governess',
    'than',
    'a',
    'friend',
    'very',
    'fond',
    'of',
    'both',
    'daughters',
    'but',
    'particularly',
    'of',
    'emma']
    ```

# Word Embeddings with word2vec <a id="word_2_vec"></a> 
After cleaning the text corpus one can start with word2vec operations.

- word2vec is a common approach for word representations.
- Besides word2vec, you could use [GloVe](https://nlp.stanford.edu/projects/glove/) - Global vectors for word representation 

## Principal idea of word2vec <a id="idea_word2vec"></a> 
- The meaning of a word can be extracted from the surrounding context.
- word2vec is an unsupervised learning technique. 
    - --> No labeled data is needed. 
    - --> Every natural language text can be used as input.
- Two model architectures:
    - **Skip-gram architecture (SG)**: Predict context words based on a target word
    - **Continuous bag of words (CBOW)**: Predict target word based on surrounding context words
- Example: "you shall know a **word** by the company it keeps" 
    - Target word: **word**
    - Window size: 3
    - SG method: Try to predict **shall know a** and **by the company** based on target **word**
    - CBOW method: Try to predict **word** by the context **shall know a** and **by the company**

### CBOW in more detail:
- Create a **Bag of words** from context words.
    - Position (order) of context words does not matter.
    - Thereby take all context words from the left and right window. 
- Calculate the mean of all context words which are in this bag.
- Use the mean value to calculate the target word. 
- What does **Continuous** mean?
    - Slide the target and context window continuously word by word from the first word to the last word of the text corpus.
    - In doing so, the target word will be predicted at each position by the corresponding actual context window.
    - Via Stochstic Gradient Descent (SGD) the location of words in the vector space will be shifted and gradually optimized. 

| Architecture | Predictions |  Relative strengths |
| ---          | ---         | ---                 |
| Skip-gram (SG) | context words via target word| Better for small text corpus, good representation of rare words |
| CBOW | target word via context words | faster, slightly better representation of common words |

![image10]

## Execute word2vec <a id="execute_word2vec"></a> 
- Open Jupyter Notebook ```natural_language_preprocessing```

    ### Excute word2vec with one single command:
    ```
    model = Word2Vec(sentences=clean_sents, size=64, 
                  sg=1, window=10, iter=5,
                  min_count=10, workers=4)
    
    model.save('clean_gutenberg_model.w2v')
    ```
    ### Some explanations:
    - **sentences**: A list of lists (e.g. clean_sents) as input. Inner list: words, Outer list: sentences
    - **size**: Number of dimensions in the word vector space. Hayperparameter (to be evaluated), here: 64
    - **sg**: 1 --> Skip-gram, 0 --> CBOW
    - **window**: godd starting value 10
    - **iter**: by default, gensim method iterates five times over all words, ok for small text corpus, decrease it for longer text corpus to reduce computational effort.
    - **min_count**: hyperparameter, a minimum threshold of occurences before a word will be taken up into the vector space. Minimum of 10 is a good starting point
    - **workers**: number of processor units used for training (if CPU has 8 processsor units) than 8 is the maximum value. If you reduce this number, you have capacity left for other tasks.
    - **model.save...**: To save your trained model
    ### Explore the model
    ```
    # Load your trained model
    model = gensim.models.Word2Vec.load('clean_gutenberg_model.w2v') 
    ```
    ```
    # How many words are in the vocabulary?
    len(model.wv.vocab) 

    RESULTS:
    ------------
    10329
    ```
    ```
    # Get the vector coordinates for 'dog' --> 64 dimensions
    model.wv['dog']

    RESULTS:
    ------------
    array([ 0.38401067,  0.01232518, -0.37594706, -0.00112308,  0.38663676,
        0.01287549,  0.398965  ,  0.0096426 , -0.10419296, -0.02877572,
        0.3207022 ,  0.27838793,  0.62772304,  0.34408906,  0.23356602,
        0.24557391,  0.3398472 ,  0.07168821, -0.18941355, -0.10122284,
       -0.35172758,  0.4038952 , -0.12179806,  0.096336  ,  0.00641343,
        0.02332107,  0.7743452 ,  0.03591069, -0.20103034, -0.1688079 ,
       -0.01331445, -0.29832968,  0.08522387, -0.02750671,  0.32494134,
       -0.14266558, -0.4192913 , -0.09291836, -0.23813559,  0.38258648,
        0.11036541,  0.005807  , -0.16745028,  0.34308755, -0.20224966,
       -0.77683043,  0.05146591, -0.5883941 , -0.0718769 , -0.18120563,
        0.00358319, -0.29351747,  0.153776  ,  0.48048878,  0.22479494,
        0.5465321 ,  0.29695514,  0.00986911, -0.2450937 , -0.19344331,
        0.3541134 ,  0.3426432 , -0.10496043,  0.00543602], dtype=float32)
    ```
    ```
    # Get similar words
    model.wv.most_similar('father', topn=3)

    RESULTS:
    ------------
    [('mother', 0.8257375359535217),
    ('brother', 0.7275018692016602),
    ('sister', 0.7177823781967163)]
    ```
    ```
    # Get the word which fits at least in a sequence 
    model.wv.doesnt_match("mother father sister brother dog".split())

    RESULTS:
    ------------
    'dog'
    ```
    ### Apply some arithmetics: **v<sub>father</sub> - v<sub>man</sub> + v<sub>woman</sub>** 
    ```
    # Arithmetics
    model.wv.most_similar(positive=['father', 'woman'], negative=['man']) 

    RESULTS:
    ------------
    [('mother', 0.7650133371353149),
    ('husband', 0.7556628584861755),
    ('sister', 0.7482180595397949),
    ('daughter', 0.7390402555465698),
    ('wife', 0.7284981608390808),
    ('sarah', 0.6856439113616943),
    ('daughters', 0.6652647256851196),
    ('conceived', 0.6637862920761108),
    ('rebekah', 0.6580977439880371),
    ('dearly', 0.6398962736129761)]
    ```

## Plot word vectors <a id="plot_word_vecs"></a> 
- Open Jupyter Notebook ```natural_language_preprocessing```
- Useful tool: **t-distributed stochastic neighbour embedding (t-SNE)**
- Example: Projection of 64 dimensional word vector space down to 2 dimensions
    ### Reduce word vector dimensionality with t-SNE
    ```
    tsne = TSNE(n_components=2, n_iter=1000)
    X_2d = tsne.fit_transform(model.wv[model.wv.vocab])
    coords_df = pd.DataFrame(X_2d, columns=['x','y'])
    coords_df['token'] = model.wv.vocab.keys()
    ```
    ### Explanation
    - **n_components**: number of output dimensions 
    - **n_iter**: number of iterations over the input data 
    ### Visualize
    ```
    _ = coords_df.plot.scatter('x', 'y', figsize=(12,12), 
                           marker='.', s=10, alpha=0.2)
    ```
    ![image3]

    Scatter plot

- For a better visualization use **bokeh** library (see page 265)
    ```
    output_notebook()
    subset_df = coords_df.sample(n=5000)
    p = figure(plot_width=800, plot_height=800)
    _ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
    show(p)
    ```
# Receiver Operating Characteristic - ROC <a id="roc"></a> 
- Please check out my repo based on [Practical Statistics](https://github.com/ddhartma/Practical-Statistics#prec_and_recall) for further information
- ROC AUC - Receiver Operating Characteristic (AUC = area under then curve)
- Useful for binary classification
- Example: Sentiment anlysis - Positive or Negativ Review?
- Advantages:
    - ROC AUC combines two metrics to one value:
        - True-Positive-Rate
        - False-Positive-Rate
    - Qualifies the output of a binary classificator in the whole range between [0,1]. This is in contradiction to metric 'accuracy' which evaluates the performance only via a threshold (y=0.5).

## Truth table - Confusion matrix <a id="truth_table"></a> 
- Evaluation of: How confused is a model?

    ![image5]

    - <span style="color:green">**TP - True Positive**</span>: Your input belongs to class (label) 1 and the model predicts it correctly.
    - <span style="color:red">**FP - False Positive**</span>: Your input belongs to class (label) 0 and the model predicts it incorrectly. Your model is confused.
    - <span style="color:red">**FN - False Negative**</span>: Your input belongs to class (label) 1 and the model predicts it incorrectly. Your model is confused.
    - <span style="color:green">**TN - True Negative**</span>: Your input belongs to class (label) 0 and the model predicts it correctly.


## Calculating ROC-AUC-Metric <a id="roc_calc"></a> 
- Let's start with an example: 
    - Let's assume we have four inputs (e.g. images). Two of them are cats and the other two are not cats.
    - For each input, we get a prediction from the model.

    ![image6]

    - In order to calculate the ROC-AUC-metric --> make classifications based on different thresholds (here 0.3, 0.5, 0.6) as shown in the table
    - Then calculate the **True-Positive-Rate** and the **False-Positive-Rate** for each threshold as shown in the table
    - Then draw a diagram:
        1. (0, 0): lower left corner
        2. (0, 0.5) for threshold 0.6
        3. (0.5, 0.5) for threshold 0.5 
        4. (0.5, 1) for threshold 0.3
        5. (1, 1) upper right corner

    ![image7]

    - **In this plot 75% of the area are below the orange line**
    - ROC-AUC metric = 0.75

- In case of random: ROC-AUC metric = 0.5
- A perfect ROC-AUC = 1 --> TPR=1 and FPR=0 for all thresholds


# Natural Language Classification with Familiar Networks <a id="nlp_class_a"></a>  
- Let's test the theory on a binary classification model: IMDb Sentiment Analysis
- Let's predict if a film review is positive or negative.
- Let's use:
    - A neural Network with fully connceted (FC) layers
    - A Convolutional Neural NEtwork (CNN)

## Loading the IMDB Film Reviews <a id="loading_imdb"></a> 
- Open Jupyter Notebook ```dense_sentiment_classifier.ipynb```
    ### Load dependencies 
    - output_dir: destination to save model parameters after each epoch
    - epochs: number of epochs
    - batch_size: the batch size for training 
    - n_dim: number of dimensions for the word vector space
    - n_unique_words: Sort every token based on frequency and then take only the most frequent words, e.g. n_unique_words = 5000
    - n_words_to_skip: Consider that the most frequent words (like 'the', 'a', 'and' etc.) are normally stop words. Skip for example the first 50 most frequent words.
    - max_review_length: All reviews must have the same length. For example set max_review_length to 100 words. Smaller reviews will be filled with Zeros (Padding).
    - pad_type: 'pre' fill Padding zeros at the beginning of the review. 'post' fill Padding zeros at the end of the review. In case of Recurrent Neural Networks (RNN) use 'pre'.
    - trunc_type: Cutting type for longer reviews --> 'pre' or 'post'
    - n_dense: number of neurons of the dense hidden layer
    - dropout: ratio of dropped out neurons per random 
    ```
    import keras
    from keras.datasets import imdb # new! 
    from keras.preprocessing.sequence import pad_sequences #new!
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout
    from keras.layers import Embedding # new!
    from keras.callbacks import ModelCheckpoint # new! 
    import os # new! 
    from sklearn.metrics import roc_auc_score, roc_curve # new!
    import pandas as pd
    import matplotlib.pyplot as plt # new!
    %matplotlib inline
    ``` 
    ### Set hyperparameters
    ```
    # output directory name:
    output_dir = 'model_output/dense'

    # training:
    epochs = 4
    batch_size = 128

    # vector-space embedding: 
    n_dim = 64
    n_unique_words = 5000 # as per Maas et al. (2011); may not be optimal
    n_words_to_skip = 50
    max_review_length = 100
    pad_type = trunc_type = 'pre'

    # neural network architecture: 
    n_dense = 64
    dropout = 0.5
    ```
    ### Load data
    - IMDb consists of 50000 film reviews
    - Half of reviews used for training
    - Half of reviews used for model validation
    - reviews with four or less stars: NEGATIVE
    - reviews with seven or more stars: POSITIVE
    - reviews with five or six stars --> are dropped (no strong review) 
    ```
    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, 
                                                        skip_top=n_words_to_skip) 
    ```


## Examining the IMDB Data <a id="examine_imdb"></a>  
- Open Jupyter Notebook ```dense_sentiment_classifier.ipynb```
    ### Take a look at the first 6 reviews(furst two are shown)
    ```
    x_train[0:6] # 0 reserved for padding; 1 would be starting character; 2 is unknown; 3 is most common word, etc.

    RESULTS:
    ------------
    array([list([2, 2, 2, 2, 2, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 2, 173, 2, 256, 2, 2, 100, 2, 838, 112, 50, 670, 2, 2, 2, 480, 284, 2, 150, 2, 172, 112, 167, 2, 336, 385, 2, 2, 172, 4536, 1111, 2, 546, 2, 2, 447, 2, 192, 50, 2, 2, 147, 2025, 2, 2, 2, 2, 1920, 4613, 469, 2, 2, 71, 87, 2, 2, 2, 530, 2, 76, 2, 2, 1247, 2, 2, 2, 515, 2, 2, 2, 626, 2, 2, 2, 62, 386, 2, 2, 316, 2, 106, 2, 2, 2223, 2, 2, 480, 66, 3785, 2, 2, 130, 2, 2, 2, 619, 2, 2, 124, 51, 2, 135, 2, 2, 1415, 2, 2, 2, 2, 215, 2, 77, 52, 2, 2, 407, 2, 82, 2, 2, 2, 107, 117, 2, 2, 256, 2, 2, 2, 3766, 2, 723, 2, 71, 2, 530, 476, 2, 400, 317, 2, 2, 2, 2, 1029, 2, 104, 88, 2, 381, 2, 297, 98, 2, 2071, 56, 2, 141, 2, 194, 2, 2, 2, 226, 2, 2, 134, 476, 2, 480, 2, 144, 2, 2, 2, 51, 2, 2, 224, 92, 2, 104, 2, 226, 65, 2, 2, 1334, 88, 2, 2, 283, 2, 2, 4472, 113, 103, 2, 2, 2, 2, 2, 178, 2]),
       list([2, 194, 1153, 194, 2, 78, 228, 2, 2, 1463, 4369, 2, 134, 2, 2, 715, 2, 118, 1634, 2, 394, 2, 2, 119, 954, 189, 102, 2, 207, 110, 3103, 2, 2, 69, 188, 2, 2, 2, 2, 2, 249, 126, 93, 2, 114, 2, 2300, 1523, 2, 647, 2, 116, 2, 2, 2, 2, 229, 2, 340, 1322, 2, 118, 2, 2, 130, 4901, 2, 2, 1002, 2, 89, 2, 952, 2, 2, 2, 455, 2, 2, 2, 2, 1543, 1905, 398, 2, 1649, 2, 2, 2, 163, 2, 3215, 2, 2, 1153, 2, 194, 775, 2, 2, 2, 349, 2637, 148, 605, 2, 2, 2, 123, 125, 68, 2, 2, 2, 349, 165, 4362, 98, 2, 2, 228, 2, 2, 2, 1157, 2, 299, 120, 2, 120, 174, 2, 220, 175, 136, 50, 2, 4373, 228, 2, 2, 2, 656, 245, 2350, 2, 2, 2, 131, 152, 491, 2, 2, 2, 2, 1212, 2, 2, 2, 371, 78, 2, 625, 64, 1382, 2, 2, 168, 145, 2, 2, 1690, 2, 2, 2, 1355, 2, 2, 2, 52, 154, 462, 2, 89, 78, 285, 2, 145, 95]),
    ...   
    ```
    ### Check the length of the first six reviews
    ```
    for x in x_train[0:6]:
        print(len(x))

    RESULTS:
    -----------
    218
    189
    141
    550
    147
    43
    ```
    ### The Labels
    ```
    y_train[0:6] 

    RESULTS:
    ------------
    array([1, 0, 0, 1, 0, 0])
    ```
    ### Size of training and validation set
    ```
    len(x_train), len(x_valid)

    RESULTS:
    ------------
    (25000, 25000)
    ```
    ### Restoring words from index (in order to read reviews)
    ```
    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    word_index["UNK"] = 2

    print(word_index)

    RESULTS:
    ------------
    {'fawn': 34704,
    'tsukino': 52009,
    'nunnery': 52010,
    'sonja': 16819,
    'vani': 63954,
    'woods': 1411,
    'spiders': 16118,
    'hanging': 2348,
    'woody': 2292,
    'trawling': 52011,
    "hold's": 52012,
    'comically': 11310,
    'localized': 40833,
    'disobeying': 30571,
    "'royale": 52013,
    "harpo's": 40834,
    'canet': 52014,
    ...
    }
    ```
    ### The first review
    ```
    x_train[0] 

    RESULTS:
    ------------
    [2,
    2,
    2,
    2,
    2,
    530,
    973,
    1622,
    1385,
    65,
    458,
    4468,
    66,
    3941,
    2,
    173,
    2,
    256,
    ...
    ]
    ```
    ### Convert it back to readable text
    ```
    ' '.join(index_word[id] for id in x_train[0])

    RESULTS:
    ------------
    "UNK UNK UNK UNK UNK brilliant casting location scenery story direction everyone's really suited UNK part UNK played UNK UNK could UNK imagine being there robert UNK UNK UNK amazing actor UNK now UNK same being director UNK father came UNK UNK same scottish island UNK myself UNK UNK loved UNK fact there UNK UNK real connection UNK UNK UNK UNK witty remarks throughout UNK UNK were great UNK UNK UNK brilliant UNK much UNK UNK bought UNK UNK UNK soon UNK UNK UNK released UNK UNK UNK would recommend UNK UNK everyone UNK watch UNK UNK fly UNK UNK amazing really cried UNK UNK end UNK UNK UNK sad UNK UNK know what UNK say UNK UNK cry UNK UNK UNK UNK must UNK been good UNK UNK definitely UNK also UNK UNK UNK two little UNK UNK played UNK UNK UNK norman UNK paul UNK were UNK brilliant children UNK often left UNK UNK UNK UNK list UNK think because UNK stars UNK play them UNK grown up UNK such UNK big UNK UNK UNK whole UNK UNK these children UNK amazing UNK should UNK UNK UNK what UNK UNK done don't UNK think UNK whole story UNK UNK lovely because UNK UNK true UNK UNK someone's life after UNK UNK UNK UNK UNK us UNK"
    ```

## Standardizing the Length of the Reviews <a id="standardize_imdb"></a>  
- Open Jupyter Notebook ```dense_sentiment_classifier.ipynb```
    ### Padding and Truncating reviews
    ```
    x_train = pad_sequences(x_train, maxlen=max_review_length, 
                        padding=pad_type, truncating=trunc_type, value=0)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length, 
                            padding=pad_type, truncating=trunc_type, value=0)

    for x in x_train[0:6]:
        print(len(x)) 

    RESULTS:
    ------------
    100
    100
    100
    100
    100
    100            
    ```

## Dense Network <a id="dense_net"></a> 
- Open Jupyter Notebook ```dense_sentiment_classifier.ipynb```
    ### Design a fully connected neural network architecture
    - Embedding: it is a look up table with 5000 words. It will be trained via Gradient Descent. 
    - Flatten: Needed to use a fully connected layer approach
    - 1 Hidden layer
    - Output via sigmoid  
    ```
    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dropout(dropout))
    # model.add(Dense(n_dense, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid')) # mathematically equivalent to softmax with two classes
    
    model.summary() # so many parameters!

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 64)           320000    
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 6400)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                409664    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 729,729
    Trainable params: 729,729
    Non-trainable params: 0
    ```

    ### Explanation
    - **embedding_1** has 320000 parameters: **n_dim** * **n_unique_words** = 64 * 5000
    - **flatten_1** gets 6400 input values: 100 words of each review * 64 word vector dimensions
    - **dense_1** has 409664 parameters to train --> 64 hidden neurons of dense_1 times 6400 values from the flatten layer.
    - **dense_2** output layer has 65 parameters, one for activation of each neuron of the previous layer plus one Bias value.
    - IN TOTAL: 730.000 parameters to train.

    ### Compile the model
    - Use **binary_crossentropy** instead of **categorical_crossentropy** (used for multi class problems)
    - Store model parameters in **modelcheckpoint** after each epoch 
    ```
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    modelcheckpoint = ModelCheckpoint(filepath=output_dir+
                                  "/weights.{epoch:02d}.hdf5")
    ```
    ### Start Training 
    ```
    model.fit(x_train, y_train, 
          batch_size=batch_size, epochs=epochs, verbose=1, 
          validation_data=(x_valid, y_valid), 
          callbacks=[modelcheckpoint])

    RESULTS:
    ------------
    ...
    Epoch 4/4
    25000/25000 [==============================] - 2s 70us/step - loss: 0.0237 - acc: 0.9961 - val_loss: 0.5304 - val_acc: 0.8340
    ```
    ### Evaluate
    ```
    # Load weights
    model.load_weights(output_dir+"/weights.02.hdf5") # NOT zero-indexed

    # Predict 
    y_hat = model.predict_proba(x_valid)
    print(len(y_hat))
    RESULTS:
    ------------
    25000

    # Draw a histogram
    plt.hist(y_hat)
    _ = plt.axvline(x=0.5, color='orange')
    ```
    ![image8]

    - Approximately for 8000 reviews (out of 25000 --> 32%) the prediction is less than 0.1 (--> NEGATIVE review)
    - Approximately for 6500 reviews (out of 25000 --> 26%) the prediction is larger than 0.9 (--> POSITIVE review)
    - The orange line marks the **threshold** (0.5) for claassification
    
    ### Calculate ROC-AUC score:
    ```
    pct_auc = roc_auc_score(y_valid, y_hat)*100.0
    "{:0.2f}".format(pct_auc) 

    RESULTS:
    ------------
    '92.85'
    ```

## Convolutional Networks <a id="conv_net"></a> 
- Let's use CNNs to detect spatial patterns in words, e.g. 'not good'
    ### Load dependencies 
    - Same as in [Dense Network](#dense_net)
    - Additional: 
    ```
    from keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D # new! 
    ```
    ### Set hyperparameters
    - conv: directory to save model parameters 
    ```
    # output directory name:
    output_dir = 'model_output/conv'

    # training:
    epochs = 4
    batch_size = 128

    # vector-space embedding: 
    n_dim = 64
    n_unique_words = 5000 
    max_review_length = 400
    pad_type = trunc_type = 'pre'
    drop_embed = 0.2 # new!

    # convolutional layer architecture:
    n_conv = 256 # filters, a.k.a. kernels
    k_conv = 3 # kernel length

    # dense layer architecture: 
    n_dense = 256
    dropout = 0.2
    ```
    ### Explanation:
    - max_review_length = 400: In the Dense layer approach we had 100. This increase is possible as the amount of parameters is drasrtically decreased by Convolutional layers.
    - We still use an embedding layer
    - Two hidden layers:
        - one Conv1D layer: 
            - 256 filter (n_conv) 
            - kernel size: 3
            - **Language is one dimensional (dimension of time)**
            - Remember: Images are two dimensional (Conv2D)
        - one FC layeer:
            - 256 neurons
            - dropout: 20%
    ### Load and Preprocess data 
    - Same as in [Dense Network](#dense_net)
    
    ### Design neural network architecture 
    ```
    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(SpatialDropout1D(drop_embed))
    model.add(Conv1D(n_conv, k_conv, activation='relu'))
    # model.add(Conv1D(n_conv, k_conv, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.summary() 

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 400, 64)           320000    
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 400, 64)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 398, 256)          49408     
    _________________________________________________________________
    global_max_pooling1d_1 (Glob (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               65792     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 435,457
    Trainable params: 435,457
    Non-trainable params: 0
    ```
    ### Some explanations:
    - Dropout is used after the Embedding layer
    - No Flatten() approach is needed anymore --> Conv approach 
    - Conv1D layer: this layer consists of 256 filters which learn to get active if if the kernel runs over a special sequence of three tokens
    - In NLP global_max_pooling is common. This reduces the activation map from 256x398 to 256x1
    - The whole network has 435457 parameter.

    ### Compile, Train model
    - Same as in [Dense Network](#dense_net)

    ### Evaluate 

    ```
    ...

    # Draw a histogram
    plt.hist(y_hat)
    _ = plt.axvline(x=0.5, color='orange')
    ```
    ![image9] 

    Compared to the FC approach above:
    - More reviews less than 0.1 (--> NEGATIVE review)
    - More reviews larger than 0.9 (--> POSITIVE review)
    - The orange line marks the **threshold** (0.5) for claassification
    
    ### Calculate ROC-AUC score:
    ```
    pct_auc = roc_auc_score(y_valid, y_hat)*100.0
    "{:0.2f}".format(pct_auc) 

    RESULTS:
    ------------
    '96.12'
    ```
    - The CNN approach is better than the FC approach.
    - Reasons: 
        - Better recognition of word sequences like 'not good'.
        - Better transfer performance: 'not great' should have similar locations in the word vector space as 'not good'.

# Networks Designed for Sequential Data <a id="network_seq"></a> 
- The kernels of Convolutional Neural Networks are good to learn short sequences, e.g. a group of three words.
- However, in NLP word sequences are normally much longer
- Let's introduce:
    - Recurrent Neural Networks (RNNs)
    - Long Short-Term Memory Units (LSTMs)
    - Gated Recurrent Units (GRUs)
    - Attention
- Typical Applications: 1D Data Sequences like
    - Stock prices 
    - Selling price
    - Temperatures
    - Disease rates

## Recurrent Neural Networks (RNNs) <a id="rnn"></a> 
Take a look at the sentence:
**Jon and Grant are writing a book together. They have really enjoyed writing it.**

- The principal structure of a RNN is shown below

    ![image11]

    - Grey line indicates a data push from step to step.
    - Similar to FC networks, there is exactly one neuron for each input.
    - Additionally: Each recurrent modul receives information from the previous modul (collected information from previous time steps). For example: The RNN learns to associate 'They' with 'Jon and Grant'.

- Learning process:
    - Like in feedforward networks RNNs learn via Backpropagation. However, in RNNs the error is not only backpropagated to the input layer but also via all (i.e. previous) time steps of the recurrent layers. 
    - Later time steps have more influence on predictions then previous ones.

- Open Jupyter Notebook ```rnn_sentiment_classifier.ipynb```.
    ### Load dependencies
    ```
    import keras
    from keras.datasets import imdb
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D
    from keras.layers import SimpleRNN # new! 
    from keras.callbacks import ModelCheckpoint
    import os
    from sklearn.metrics import roc_auc_score 
    import matplotlib.pyplot as plt 
    %matplotlib inline
    ```
    ### Set hyperparameters
    - max_review_length: 100, for a simple RNN it could still be too excessive due to Vanishing Gradients (better with LSTMs)
    - n_rnn: 256, recurrent layer has 256 units. This allows the RNN to identify 256 unique sequences word meanings which are relevant to analyze the  sentiment of the review. This is similar to 256 convolutional kernels in a CNN model, which specialize to identify on 256 unique three-word-groups

    ```
    # output directory name:
    output_dir = 'model_output/rnn'

    # training:
    epochs = 16 # way more!
    batch_size = 128

    # vector-space embedding: 
    n_dim = 64 
    n_unique_words = 10000 
    max_review_length = 100 # lowered due to vanishing gradient over time
    pad_type = trunc_type = 'pre'
    drop_embed = 0.2 

    # RNN layer architecture:
    n_rnn = 256 
    drop_rnn = 0.2

    # dense layer architecture: 
    # n_dense = 256
    # dropout = 0.2
    ```
    ### Load data
    ```
    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words) # removed n_words_to_skip
    ```
    ### Preprocess Data
    ```
    x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    ```
    ### Design neural network architecture
    ```
    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(SpatialDropout1D(drop_embed))
    model.add(SimpleRNN(n_rnn, dropout=drop_rnn))
    # model.add(Dense(n_dense, activation='relu')) # typically don't see top dense layer in NLP like in 
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()  

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 64)           640000    
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 100, 64)           0         
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 256)               82176     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 722,433
    Trainable params: 722,433
    Non-trainable params: 0
    ```
    ### Compile model
    ```
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ```
    ### Train
    ```
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

    RESULTS:
    ------------
    Epoch 16/16
    25000/25000 [==============================] - 17s 672us/step - loss: 0.4012 - acc: 0.8309 - val_loss: 0.5920 - val_acc: 0.7510
    ```
    ### Evaluate
    ```
    model.load_weights(output_dir+"/weights.07.hdf5") 
    y_hat = model.predict_proba(x_valid)

    plt.hist(y_hat)
    _ = plt.axvline(x=0.5, color='orange')
    ```

    ![image12]

    ### ROC AUC:
    ```
    "{:0.2f}".format(roc_auc_score(y_valid, y_hat)*100.0)

    RESULTS:
    ------------
    '84.94'
    ```
    ### Reason for bad performance:
    - It is believed that Vanishing Gradient effects due to a long max_review_length=100 are too strong.
    - Maybe one should reduce (not checked) max_review_length to 10.


## Long Short-Term Memory Units (LSTMs) <a id="lstm"></a> 
- Interesting article: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Simple RNNs are sufficient for short sequences in the range of 10 time steps. 
- Use LSTMs for longer sequences.
### Principal structure:
- The principal structure is identical to simple RNNs. It is a special kind of RNN, **capable of learning long-term dependencies**. LSTMs are explicitly designed to avoid the long-term dependency problem.
- They were introduced by Hochreiter & Schmidhuber (1997).
- LSTMs get inputs 
    - from the sequence and
    - from previous time steps
- The difference:
    - RNN: 
        - In each cell of the recurrent layer there is only on activation function like a Tanh-function. 
        - They transport only **one type of information** through the whole network:
            - hidden state --> short-term memory
    - LSTM: 
        - More complex structure of gating functions. Instead of having a single neural network layer, there are four, interacting in a very special way. - They tranport two types of information:
            - hidden state --> short-term memory 
            - cell state --> long-term memory

    ![image14]

- **The Core Idea Behind LSTMs**: 
    - The key to LSTMs is the **cell state c<sub>t</sub>**, the horizontal line running through the top of the diagram. Runs from cell to cell. There is no nonlinear activation function. Only two linear transformations (one **addition** and one **multiplication** operation). Information will be passed from cell to cell. 
    - There is a Gate which decides if something will be added to the cell state or not. It is composed out of a **sigmoid neural net layer** and a **pointwise multiplication** operation. 
    - The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”
    - An LSTM has three of these gates, to protect and control the cell state.
    - Adding information: New information is a concatenation of **new input x<sub>t</sub>** (actual time step) and **hidden state h<sub>t-1</sub>**  (previuos time step). The **hidden state** is also the **output** of the cell (it is just a copy).
    
    ![image13]    

- **For a better intuition**:
    - The **cell state** makes it possible to remain information about the length of the sequence through each time step in a particular LSTM cell. It is the LSTM's **long-term memory**.
    - The **hidden state** is analogous to the recurrent connections in a simple RNN and represents the LSTM's **short-term memory**.
    - Each **LSTM modul** represents **a specific point in the sequence** of data (e.g. a token out of a text document).
    - At each time step several **decisions will be made (via Sigmoid Gates)** if the information at this actual time step in the sequence is relevant for the local (hidden state) and global (cell state) context.
    - **First two Sigmoid Gates decide** if the information from the actual time step is **relevant for the global context** (cell state) and how to combine the information into the stream.
    - **Last Sigmoid Gate determines** if the information from the actual time step is **relevant for the local context** (hidden state / output).

- Open Jupyter Notebook ```lstm_sentiment_classifier.ipynb```.
    ### Load dependencies
    ``` 
    import keras
    from keras.datasets import imdb
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D
    from keras.layers import LSTM # new! 
    from keras.callbacks import ModelCheckpoint
    import os
    from sklearn.metrics import roc_auc_score 
    import matplotlib.pyplot as plt 
    %matplotlib inline
    ```
    ### Set hyperparameters
    - Same hyperparameters as for RNN (see above), exception: epochs=4 (LSTM reacts more sensitive to overfitting than a simple RNN)
    ```
    # output directory name:
    output_dir = 'model_output/LSTM'

    # training:
    epochs = 4
    batch_size = 128

    # vector-space embedding: 
    n_dim = 64 
    n_unique_words = 10000 
    max_review_length = 100 
    pad_type = trunc_type = 'pre'
    drop_embed = 0.2 

    # LSTM layer architecture:
    n_lstm = 256 
    drop_lstm = 0.2

    # dense layer architecture: 
    # n_dense = 256
    # dropout = 0.2
    ```
    ### Load data
    ```
    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words) # removed n_words_to_skip
    ```
    ### Preprocess data
    ```
    x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    ```
    ### Design neural network architecture
    ```
    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(SpatialDropout1D(drop_embed))
    model.add(LSTM(n_lstm, dropout=drop_lstm))
    # model.add(Dense(n_dense, activation='relu')) 
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.summary() 

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 64)           640000    
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 100, 64)           0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 256)               328704    
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 968,961
    Trainable params: 968,961
    Non-trainable params: 0
    ```
    ### Compile model
    ```
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ```
    ### Train!
    ```
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

    RESULTS:
    ------------
    Epoch 4/4
    25000/25000 [==============================] - 49s 2ms/step - loss: 0.2018 - acc: 0.9235 - val_loss: 0.3690 - val_acc: 0.8454
    ```
    ### Evaluate
    ```
    model.load_weights(output_dir+"/weights.02.hdf5") 
    y_hat = model.predict_proba(x_valid)
    plt.hist(y_hat)
    _ = plt.axvline(x=0.5, color='orange')
    ```

    ![image15]
    ```

    "{:0.2f}".format(roc_auc_score(y_valid, y_hat)*100.0)

    RESULTS:
    ------------
    '92.76'
    ```

## Bidirectional LSTMs (Bi-LSTMs) <a id="bidirec_lstm"></a> 
- Bidirectional LSTMs enable a Backpropagation in both directions (backward and forward) of a one dimensional input:
    - a normal Backpropagation (e.g. from the end of a text review to the beginning)
    - a 'backward' Backpropagation (e.g. from the beginning to the end of a text review)
- Bi-LSTMs learn to interprete patterns before and after a token.
- Consequences: Doubling of computational effort but increase in accuracy.
- Bi-LSTMs are popular in modern NLP applications. 
- Open Jupyter Notebook ```bi_lstm_sentiment_classifier.ipynb```.
    ### Adapt LSTM code
    - Use the LSTM code from above.
    - Only change: Put the LSTM layer in a Bidirectional-Wrapper.
    ```
    from keras.layers import LSTM
    from keras.layers.wrappers import Bidirectional # new! 

    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(SpatialDropout1D(drop_embed))
    model.add(Bidirectional(LSTM(n_lstm, dropout=drop_lstm)))
    model.add(Dense(1, activation='sigmoid'))
    ```
    ### Result:
    - Higher validation accuracy of **86%** (simple LSTM: val_acc=85%)
    - Higher ROC-AUC: 93.5% (simple LSTM: 92.8%)
    - In this repo: Bi-LSTM is the 2nd best architecture behind the Convolutional architecture.

## Stacked Recurrent Models <a id="stacked_rnn"></a>  
- Check additional information [What are the advantages of stacking multiple LSTMs?](https://stats.stackexchange.com/questions/163304/what-are-the-advantages-of-stacking-multiple-lstms)
-
- Stacking of layers for RNN type architectures is not as simple as for FC or CNN networks.
- Additional layers (as usual) enable learning of more complex representations

    ![image16]

- Open Jupyter Notebook ```stacked_bi_lstm_sentiment_classifier.ipynb```
    ### Adapt LSTM code
    - Use the Bi-LSTM code from above.
    - Only Changes: Add a 2nd Bidirectional-Wrapper and set return_sequences=True
    ```
    from keras.layers import LSTM
    from keras.layers.wrappers import Bidirectional # new! 

    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(SpatialDropout1D(drop_embed))
    model.add(Bidirectional(LSTM(n_lstm_1, dropout=drop_lstm, 
                                return_sequences=True))) 
    model.add(Bidirectional(LSTM(n_lstm_2, dropout=drop_lstm)))
    model.add(Dense(1, activation='sigmoid'))
    ```
    ### Result:
    - Higher validation accuracy of **88%** (Bi-LSTM: val_acc=86%)
    - Higher ROC-AUC: 95.0% (Bi-LSTM: 93.5%)
    - In this repo: Bi-LSTM is the 2nd best architecture behind the Convolutional architecture.

## seq2eq and Attention <a id="seq2seq_attentiom"></a>  
### Seq2Seq
- Sequence-to-Sequence models consist of an **RNN-encoder**, a **context vector** and an **RNN-decoder** 
- **Encoder** takes an input sequence (letters, words, images, etc.) 
- **Decoder** generates output 
- **Context Vector**: Vector of numbers, which were encoded by the encoder. Typical embedding dimensions: 256 or 512. This can lead to overfitting if phrases are short (solution: Attention).

- Useful for:
    - Neural Machine translation (NMT), e.g. Google Translate
    - Questions and answers

    ![image17]

### Attention
- Context vector consists of all hidden states (here: h1, h2, h3) and not only of the last one
- Benefit: flexible context size
- --> Longer sequences can have longer context vectors, to capture the information from the input.
- --> Better correlation between words of input sequence and the different hidden: states h1 --> 1st word, h2 --> 2nd word, etc.
- --> Last hidden state has (a bit of) information about the preceeded sequence.

    ![image18]

### The Whole Attention Process
1. **The Embedding process** 

    ![image19]

2. **The Encoding Process**

    ![image20]

3. **) The Decoding Phase**: 
    
    1. **Seq2Seq w/o Attention**: Only the last hidden state will be analyzed

    ![image21]

    2. **Seq-2eq with Attention**: Create a new context vector via scoring

    ![image22]

    - **The Scoring Process**

    ![image23]

- Annotation: There are two Scoring approaches for Attention (see Udacity - Deep Learning for more information)

    - Additive Attention

    ![image24]

    - Multiplicative Attention

    ![image25]


## Transfer Learning in NLP <a id="transfer_learn_nlp"></a>  
- Transfer Learning is not only a key tool for Machine Vision, but also for NLP.
- Some Open Source resources: 
    - ULMFiT (Universal Language Model Fine-Tuning)
        - [Understanding ULMFiT — The Shift Towards Transfer Learning in NLP](https://towardsdatascience.com/understanding-ulmfit-and-elmo-the-shift-towards-transfer-learning-in-nlp-b5d8e2e3f664)
        - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
        - [Universal Language Model Fine-tuning](https://paperswithcode.com/method/ulmfit)
    - ELMo (Embeddings from Language Models)
        - [ELMo weights](https://allennlp.org/elmo)
        - [A Step-by-Step NLP Guide to Learn ELMo for Extracting Features from Text](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)
    - BERT (Bi-Directional Encoder Representation from Transformers)
        - [BERT Explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
        - [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

# Non-Sequential Architectures: The Keras Functional API <a id="non_seq_architect"></a>
- Istead Using n the Keras class **Sequential()** use the **Model** class of Keras
- Non sequential model allow infinite network architectures and more creative networks. However, they are more complex.
- Example below: Three ConvNets in parallel, which take a word vector from the embedding layer. 
- One ConvNet is specialized on the recognition of word vector pairs.
- One ConvNet is specialized on the recognition of three word vectors.
- One ConvNet is specialized on the recognition of four word vectors.
- Then all three streams will be concated and transported to two fully connected layers and a sgmoid layer at the end.
- Check out [multi_convnet_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/multi_convnet_sentiment_classifier.ipynb) for implementation.

    ![image26]

## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a id="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Matrix-Math-with-Numpy.git
```

- Change Directory
```
$ cd Matrix-Math-with-Numpy
```

- Create a new Python environment, e.g. matrix_op. Inside Git Bash (Terminal) write:
```
$ conda create --id matrix_op
```

- Activate the installed environment via
```
$ conda activate matrix_op
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Important web sites - Deep Learning
* Deep Learning - illustriert - [GitHub Repo](https://github.com/the-deep-learners/deep-learning-illustrated)
* Jason Yosinski - [Visualize what kernels are doing](https://www.youtube.com/watch?v=AgkfIQ4IGaM)

Further Resources
* Read about the [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) model. Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found [here](https://arxiv.org/pdf/1609.03499.pdf).
* Learn about CNNs [for text classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). You might like to sign up for the author's [Deep Learning Newsletter!](https://www.getrevue.co/profile/wildml)
* Read about Facebook's novel [CNN approach for language translation](https://engineering.fb.com/2017/05/09/ml-applications/a-novel-approach-to-neural-machine-translation/) that achieves state-of-the-art accuracy at nine times the speed of RNN models.
* Play [Atari games with a CNN and reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning). If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).
* Play [pictionary](https://quickdraw.withgoogle.com/#) with a CNN! Also check out all of the other cool implementations on the [A.I. Experiments](https://experiments.withgoogle.com/collection/ai) website. Be sure not to miss [AutoDraw](https://www.autodraw.com/)!
* Read more about [AlphaGo]. Check out [this article](https://www.technologyreview.com/2017/04/28/106009/finding-solace-in-defeat-by-artificial-intelligence/), which asks the question: If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?
* Check out these really cool videos with drones that are powered by CNNs.
    - Here's an interview with a startup - [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y).
    - Outdoor autonomous navigation is typically accomplished through the use of the [global positioning system (GPS)](www.droneomega.com/gps-drone-navigation-works/), but here's a demo with a CNN-powered [autonomous drone](https://www.youtube.com/watch?v=wSFYOw4VIYY).

* If you're excited about using CNNs in self-driving cars, you're encouraged to check out:
    - Udacity [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), where we classify signs in the German Traffic Sign dataset in this project.
    - Udacity [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t), where we classify house numbers from the Street View House Numbers dataset in this project.
    - This series of blog posts that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.

* Check out some additional applications not mentioned in the video.
    - Some of the world's most famous paintings have been [turned into 3D](https://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1) for the visually impaired. Although the article does not mention how this was done, we note that it is possible to use a CNN to [predict depth](https://cs.nyu.edu/~deigen/depth/) from a single image.
    - Check out [this research](https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
    - CNNs are used to [save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
    - An app called [FaceApp](https://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.

Recent publications
* [R. Girshick et al, arXiv:1311.2524v5, 2014, Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
* [R. Girshick et al., arXiv:1504.08083v2, 2015, Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
* [Shaoqing Ren et al., arXiv:1506.01497v3, 2016, Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
* [J. Redmon et al., 2016, arXiv:1506.02640v5, You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640v5.pdf)
* [Fei-Fei Li et al. 2017, Lecture 11:Detection and Segmentation](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
