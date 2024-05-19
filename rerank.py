import streamlit as st
import pandas as pd
from io import BytesIO
import voyageai
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')

def truncate_tokens(text, max_tokens=500):
    tokens = word_tokenize(text)
    truncated = ' '.join(tokens[:max_tokens])
    return truncated

def rerank_column(api_key, query, documents):
    vo = voyageai.Client(api_key=api_key)
    reranking = vo.rerank(query, documents, model="rerank-lite-1")
    return reranking.results

def main():
    st.title("Excel Column Reranker")
    api_key = st.sidebar.text_input("Enter your Voyage AI API key", type="password")

    # Step 1: Upload an Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Step 2: Input a search query
        query = st.text_input("Enter a search query")

        # Step 3: Select a column to rerank
        column = st.selectbox("Select a column to rerank", df.columns)

        if st.button("Rerank"):
            if api_key:
                documents = df[column].astype(str).tolist()
                truncated_documents = [truncate_tokens(doc) for doc in documents]
                results = rerank_column(api_key, query, truncated_documents)

                # Step 4: Show the results and rank the table based on the returned score
                ranked_indices = [r.index for r in results]
                ranked_scores = [r.relevance_score for r in results]
                df['relevance_score'] = pd.Series(ranked_scores, index=ranked_indices)
                df_sorted = df.sort_values(by='relevance_score', ascending=False)

                st.write(df_sorted)

                # Step 5: Download the file
                output = BytesIO()
                df_sorted.to_excel(output, index=False, engine='xlsxwriter')
                output.seek(0)
                st.download_button(
                    label="Download reranked Excel file",
                    data=output,
                    file_name="reranked.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Please enter your Voyage AI API key.")

if __name__ == "__main__":
    main()
