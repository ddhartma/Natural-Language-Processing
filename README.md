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
