from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

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

temp_vectorizer = TfidfVectorizer(binary=True, use_idf=False)
temp_vectorizer.fit(documents_df['processed_text'].tolist())


def calculate_similarity_vsm_query(query_text, td_matrix, vectorizer, documents_df):
    processed_query = basic_preprocess_text(query_text)
    query_vector = vectorizer.transform([processed_query])
    cosine_scores = cosine_similarity(query_vector, td_matrix).flatten()
    results_df = pd.DataFrame({
        'doc_id': documents_df['doc_id'], # Use doc_ids from the original DataFrame
        'text': documents_df['text'],     # Include original text for context
        'similarity_score': cosine_scores
    })

    results_df = results_df.sort_values(by='similarity_score', ascending=False)

    return results_df


user_query = input("Enter your search query for VSM: ")

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

vsm_results = calculate_similarity_vsm_query(user_query, td_matrix, temp_vectorizer, documents_df)

print(f"\nVSM Results for query: '{user_query}'")
#display(vsm_results)
print(vsm_results)

def calculate_precision_at_k(results_df, relevant_doc_ids, k):
    if k <= 0:
        return 0.0
    top_k_docs = results_df.head(k)
    retrieved_relevant_count = sum(1 for doc_id in top_k_docs['doc_id'] if doc_id in relevant_doc_ids)
    return retrieved_relevant_count / k if k > 0 else 0.0

def calculate_pr_curve_points(results_df, relevant_doc_ids):
    total_relevant = len(relevant_doc_ids)
    if total_relevant == 0:
        return [(0.0, 1.0)]

    retrieved_count = 0
    relevant_retrieved_count = 0
    pr_points = []

    pr_points.append((0.0, 1.0))

    for index, row in results_df.iterrows():
        retrieved_count += 1
        doc_id = row['doc_id']

        if doc_id in relevant_doc_ids:
            relevant_retrieved_count += 1

        precision = relevant_retrieved_count / retrieved_count
        recall = relevant_retrieved_count / total_relevant

        pr_points.append((recall, precision))

    if pr_points[-1][0] < 1.0:
         pr_points.append((1.0, pr_points[-1][1]))


    return pr_points

relevant_docs_for_query = {2, 3} 

if 'vsm_results' in locals() and not vsm_results.empty:
    print("--- Evaluating VSM Results ---")
    k_values = [1, 2, 3] 
    for k in k_values:
        precision_at_k_vsm = calculate_precision_at_k(vsm_results, relevant_docs_for_query, k)
        print(f"Precision @ {k} (VSM): {precision_at_k_vsm:.2f}")

    pr_points_vsm = calculate_pr_curve_points(vsm_results, relevant_docs_for_query)
    print("\nPR Curve Points (VSM):", pr_points_vsm)

    plt.figure()
    plt.plot([p[0] for p in pr_points_vsm], [p[1] for p in pr_points_vsm], marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (VSM)')
    plt.grid(True)
    plt.show()

else:
    print("VSM results (vsm_results DataFrame) not found. Please run VSM querying first.")

print("-" * 20) 
