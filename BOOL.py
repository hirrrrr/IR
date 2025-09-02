import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def basic_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = ['the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that']
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


def read_and_preprocess_documents(directory_path):
    """
    Reads text files from a directory, stores content in a DataFrame,
    assigns a document ID, and applies preprocessing.
    """
    documents_list = []
    doc_id_counter = 1

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents_list.append({
                    'doc_id': doc_id_counter,
                    'filename': filename,
                    'text': content
                })
                doc_id_counter += 1

    # Create DataFrame
    df = pd.DataFrame(documents_list)

    # Apply preprocessing
    df['processed_text'] = df['text'].apply(basic_preprocess_text)

    return df

documents_directory = 'documents'  # Path to your directory with txt files
documents_df = read_and_preprocess_documents(documents_directory)

print("\nFinal DataFrame with Processed Text:")
print(documents_df)

def build_inverted_index(documents_df):
    inverted_index = defaultdict(list)
    # Iterate through the DataFrame rows
    for index, row in documents_df.iterrows():
        doc_id = row['doc_id']
        doc_text = row['text']
        terms = row['processed_text'].split()
        for term in set(terms):
            inverted_index[term].append(doc_id)
    return dict(inverted_index)
inverted_index = build_inverted_index(documents_df)

def process_boolean_query_complex(index, query_str):
    query_str_upper = query_str.upper().strip()

    if " NOT " in query_str_upper:
        parts = query_str_upper.split(" NOT ")
        if len(parts) != 2:
            return "Error: Invalid NOT query format"

        pos_part = parts[0]
        neg_part = parts[1]

        pos_result = process_boolean_query_complex(index, pos_part)
        neg_result = process_boolean_query_complex(index, neg_part)

        return sorted(list(set(pos_result) - set(neg_result)))

    or_parts = [part.strip() for part in query_str_upper.split(" OR ")]
    result_docs = set()
    for or_part in or_parts:
        and_parts = [part.strip() for part in or_part.split(" AND ")]

        if and_parts: 
             current_docs = set(index.get(basic_preprocess_text(and_parts[0]), []))
        else:
             current_docs = set()


        for and_part in and_parts[1:]:
            current_docs &= set(index.get(basic_preprocess_text(and_part), []))

        result_docs |= current_docs

    return sorted(list(result_docs))

query1 = "information AND retrieval"
print(f"\nQuery: {query1}")
print("Result:", process_boolean_query_complex(inverted_index, query1))

query2 = "vector OR probabilistic"
print(f"\nQuery: {query2}")
print("Result:", process_boolean_query_complex(inverted_index, query2))

query3 = "document NOT space"
print(f"\nQuery: {query3}")
print("Result:", process_boolean_query_complex(inverted_index, query3))

data = {
    'document': [
        "information retrieval is a key topic.",
        "query depends on the model used.",
        "this is a very different document.",
        "another document about information retrieval.",
        "this is not a relevant document."
    ],
    'relevance': [1, 1, 0, 1, 0] # Ground truth: 1 for relevant, 0 for irrelevant
}
df = pd.DataFrame(data)

query_term = "information retrieval"
df['retrieved'] = df['document'].apply(lambda x: 1 if query_term in x else 0)

# Step 3: Calculate the evaluation metrics
y_true = df['relevance']
y_pred = df['retrieved']

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Simulated DataFrame with Retrieval Results:")
print(df)
print("\n--- Evaluation Metrics ---")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

def build_td_matrix(documents_df):
    processed_docs = documents_df['processed_text'].tolist()
    vectorizer = TfidfVectorizer(binary=True, use_idf=False)
    td_matrix = vectorizer.fit_transform(processed_docs)
    terms = vectorizer.get_feature_names_out()
    doc_ids = documents_df['doc_id'].tolist()

    return pd.DataFrame(td_matrix.toarray(), index=doc_ids, columns=terms)

td_matrix = build_td_matrix(documents_df)
print("\nTerm-Document Matrix built from DataFrame:")
#display(td_matrix) 
print(td_matrix)
