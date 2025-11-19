import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11


def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()

        ##################
        # your code here #
        ##################
        
        # add edges based on sliding window:
        for i in range(len(doc)):
            word_str = doc[i]
            
            if word_str in vocab:
                # 1. Use the string as the identifier (w1_id for NetworkX)
                # 2. Use the integer ID as the label attribute (label=vocab[word_str])
                w1_id = word_str 
                w1_label = vocab[word_str]
                
                if not G.has_node(w1_id):
                    # Add node with the integer label stored as an attribute
                    G.add_node(w1_id, label=w1_label) 

                for j in range(1, window_size):
                    if i + j < len(doc):
                        w2_str = doc[i+j]
                        
                        if w2_str in vocab and w1_id != w2_str:
                            w2_id = w2_str
                            w2_label = vocab[w2_str]
                            
                            if not G.has_node(w2_id):
                                G.add_node(w2_id, label=w2_label)
                                
                            G.add_edge(w1_id, w2_id)

        # Handle empty graphs to prevent downstream GraKeL errors
        if len(G.nodes()) == 0:
            G.add_node("dummy_node", label=-1) 
            
        graphs.append(G)
    
    return graphs

# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
# nx.draw_networkx(G_train_nx[3], with_labels=True)
# plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations
G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(
    n_iter=1,
    base_graph_kernel=VertexHistogram,
    normalize = False,
    )# your code here #

# Construct kernel matrices
K_train = gk.fit_transform(G_train) # your code here #
K_test = gk.transform(G_test)# your code here #

#Task 13

# Train an SVM classifier and make predictions


##################
# your code here #
##################


clf = SVC(kernel="precomputed") 
clf.fit(K_train, y_train)
# Predict
y_pred = clf.predict(K_test)


# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14


##################
# your code here #
##################

from grakel.kernels import ShortestPath, RandomWalk

# Define a dictionary of kernels to experiment with.
kernel_experiments = {
    "Vertex Histogram (Baseline)": VertexHistogram(normalize=True),
    
    "WL (h=2)": WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram),
    
    "WL (h=4)": WeisfeilerLehman(n_iter=4, normalize=True, base_graph_kernel=VertexHistogram),
    
    # --- COMPUTATIONALLY EXPENSIVE KERNELS ---
    # I have commented this out because your graphs contain the full vocabulary 
    # as nodes (dense graphs), making the O(N^4) Shortest Path calculation extremely slow.
    # To run it, simply remove the '#' at the start of the line below:
    
    "Shortest Path": ShortestPath(normalize=True, with_labels=True),

    # "Random Walk (lambda=0.1)": RandomWalk(
    #     kernel_type='geometric', 
    #     lamda=0.1, 
    #     normalize=True
    # ),
}

results = {}

for name, kernel in kernel_experiments.items():
    print(f"Running experiment: {name}...")
    
    try:
        # 1. Fit and Transform on Training Data
        K_train_exp = kernel.fit_transform(G_train)
        
        # 2. Transform Test Data
        K_test_exp = kernel.transform(G_test)
        
        # 3. Train SVM
        clf_exp = SVC(kernel='precomputed')
        clf_exp.fit(K_train_exp, y_train)
        
        # 4. Predict and Evaluate
        y_pred_exp = clf_exp.predict(K_test_exp)
        acc = accuracy_score(y_test, y_pred_exp)
        
        results[name] = acc
        print(f"   -> Accuracy: {acc:.4f}\n")
        
    except Exception as e:
        print(f"   -> Failed to run {name}: {e}\n")

# Display final comparison table
print("-" * 45)
print(f"{'Kernel Name':<30} | {'Accuracy':<10}")
print("-" * 45)
for name, acc in results.items():
    print(f"{name:<30} | {acc:.4f}")
print("-" * 45)