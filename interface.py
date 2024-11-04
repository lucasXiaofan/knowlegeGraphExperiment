import streamlit as st

from query_processing import rewrite_query
from openai import OpenAI
import os
import json
from knowledge_base_building import create_neo4j_node, combined_vector_search,kg
from datetime import datetime



# Initialize OpenAI client
client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )
embedding_client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"])

def save_search_result(query, results, comments=""):
    """
    Save search results to JSON file with datetime handling
    """
    filename = "search_history.json"
    
    # Create new result entry with serializable datetime
    new_result = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "comments": comments,
        "results": []
    }
    
    # Process results to ensure all values are JSON serializable
    for result in results[:2]:
        serializable_result = {
            "score": float(result['score']),  # Ensure score is float
            "source": result['source'],
            "title": result['title'],
            "content": result['content'],
            "url": result['url'],
            "knowledge_id": result['knowledge_id'],
            "creation_date": str(result['creation_date'])
        }
        new_result["results"].append(serializable_result)
    
    try:
        # Load existing results or create empty list
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except json.JSONDecodeError:
                # If file is empty or invalid, start with empty list
                all_results = []
        else:
            all_results = []
        
        # Append new result
        all_results.append(new_result)
        
        # Save updated results
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return True, f"Result saved to {filename}"
    except Exception as e:
        return False, f"Error saving result: {str(e)}"

def embedding_experiment():
    st.title("Embedding Experiment")
    
    # Create two text areas for input
    text1 = st.text_area("Enter first text:", height=100)
    text2 = st.text_area("Enter second text:", height=100)
    
    # Initialize session state for similarity result if not exists
    if 'current_similarity' not in st.session_state:
        st.session_state.current_similarity = None
    
    if st.button("Calculate Similarity"):
        if text1 and text2:
            with st.spinner("Calculating embeddings..."):
                try:
                    # Get embeddings
                    response1 = embedding_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text1
                    )
                    response2 = embedding_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text2
                    )
                    
                    # Calculate cosine similarity
                    import numpy as np
                    embedding1 = np.array(response1.data[0].embedding)
                    embedding2 = np.array(response2.data[0].embedding)
                    
                    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                    
                    # Store result in session state
                    st.session_state.current_similarity = {
                        "text1": text1,
                        "text2": text2,
                        "similarity": similarity,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Display result
                    st.success(f"Cosine Similarity: {similarity:.4f}")
                except Exception as e:
                    st.error(f"Error calculating similarity: {str(e)}")
        else:
            st.warning("Please enter both texts to compare.")
    
    # Save button (only show if there's a result to save)
    if st.session_state.current_similarity is not None:
        if st.button("Save Result"):
            try:
                filename = "embedding_results.json"
                
                # Load existing results or create empty list
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                else:
                    results = []
                
                # Append new result
                results.append(st.session_state.current_similarity)
                
                # Save updated results
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                st.success(f"Result saved to {filename}")
            except Exception as e:
                st.error(f"Error saving result: {str(e)}")

    # Display saved results
    if os.path.exists("embedding_results.json"):
        with st.expander("View Saved Results"):
            try:
                with open("embedding_results.json", 'r', encoding='utf-8') as f:
                    saved_results = json.load(f)
                
                for idx, result in enumerate(saved_results, 1):
                    st.markdown(f"""
                    **Result {idx}**
                    - Text 1: {result['text1']}
                    - Text 2: {result['text2']}
                    - Similarity: {result['similarity']:.4f}
                    - Timestamp: {result['timestamp']}
                    ---
                    """)
            except Exception as e:
                st.error(f"Error loading saved results: {str(e)}")



def display_search_results(results):
    if results:
        st.subheader(f"Found {len(results)} Results")
        for idx, result in enumerate(results, 1):
            # Create a title based on the source
            if result['source'] == 'article':
                title = result['title']
                content_preview = f"**Content Preview:**\n{result['content'][:500]}..."
                url_section = f"**URL:** [{result['url']}]({result['url']})"
            else:  # knowledge_base
                title = f"Knowledge Base Entry ({result['knowledge_id'][:8]}...)"
                content_preview = f"**Content:**\n{result['content']}"
                url_section = f"**Created:** {result['creation_date']}"

            with st.expander(f"Result {idx}: {title} (Score: {result['score']:.3f})"):
                st.markdown("---")
                st.markdown(f"**Source:** {result['source'].title()}")
                st.markdown(content_preview)
                st.markdown(url_section)
                st.markdown(f"**Similarity Score:** {result['score']:.3f}")
    else:
        st.warning("No results found.")

def main():
    # Update tab selection
    tab1, tab2, tab3 = st.tabs(["Knowledge Base Search", "Embedding Experiment", "Create Neo4j Node"])
    
    with tab1:
        # Initialize session states
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        if 'show_history' not in st.session_state:
            st.session_state.show_history = False
        
        st.title("Knowledge Base Search")
        
        question = st.text_area("Enter your search query:", height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        
        with col2:
            use_rewrite = st.radio(
                "Query Processing:",
                ["Original Query", "Optimized Query"],
                captions=["Use query as-is", "Use AI to optimize query"]
            )

        if st.button("Search"):
            if question:
                if use_rewrite == "Optimized Query":
                    with st.spinner("Optimizing your query..."):
                        rewritten_query = rewrite_query(question, client)
                        st.info(f"""
                        **Original Query:** {question}
                        **Optimized Query:** {rewritten_query}
                        """)
                    search_query = rewritten_query
                else:
                    search_query = question

                with st.spinner("Searching..."):
                    results = combined_vector_search(
                        question=search_query,
                        top_k=top_k
                    )
                    # Store results and query in session state
                    st.session_state.current_results = {
                        'query': search_query,
                        'results': results
                    }
                    display_search_results(results)
            else:
                st.warning("Please enter a search query.")

        # Only show save options if we have results
        if st.session_state.current_results is not None:
            st.divider()
            st.subheader("Save Search Results")
            
            comments = st.text_area(
                "Add comments about this search (optional):",
                height=100,
                help="Add any notes or observations about the search results"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Results"):
                    success, message = save_search_result(
                        query=st.session_state.current_results['query'],
                        results=st.session_state.current_results['results'],
                        comments=comments
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col2:
                if st.button("Toggle History View"):
                    st.session_state.show_history = not st.session_state.show_history

        # Show history if toggled
        if st.session_state.show_history:
            if os.path.exists("search_history.json"):
                with st.expander("Search History", expanded=True):
                    try:
                        with open("search_history.json", 'r', encoding='utf-8') as f:
                            history = json.load(f)
                        
                        for idx, entry in enumerate(reversed(history), 1):
                            st.markdown(f"""
                            #### Search {idx}
                            **Query:** {entry['query']}
                            **Time:** {entry['timestamp']}
                            **Comments:** {entry['comments'] or 'No comments'}
                            
                            **Results:**
                            """)
                            
                            for r_idx, result in enumerate(entry['results'], 1):
                                with st.expander(f"Result {r_idx}: {result['title']} (Score: {result['score']:.3f})"):
                                    st.markdown(f"**Source:** {result['source'].title()}")
                                    st.markdown(f"**Content Preview:**\n{result['content'][:500]}...")
                                    if result['url']:
                                        st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                                    if result['knowledge_id']:
                                        st.markdown(f"**Knowledge ID:** {result['knowledge_id']}")
                                    if result['creation_date']:
                                        st.markdown(f"**Created:** {result['creation_date']}")
                            
                            st.divider()
                            
                    except Exception as e:
                        st.error(f"Error loading search history: {str(e)}")
            else:
                st.info("No search history found")

        # Sidebar with information
        with st.sidebar:
            st.subheader("About")
            st.markdown("""
            ### How to use:
            1. Enter your search query
            2. Choose search type (Main Content/Use Cases)
            3. Adjust number of results
            4. Choose query processing method:
               - Original Query: Uses your query as-is
               - Optimized Query: Uses AI to make your query more focused
            5. Save interesting search results with comments
            
            The search uses vector similarity to find the most relevant articles.
            """)
            
            # Database statistics
            st.subheader("Database Statistics")
            try:
                article_count = kg.query("""
                    MATCH (a:Article)
                    RETURN COUNT(a) as count
                """)[0]['count']
                
                embedded_count = kg.query("""
                    MATCH (a:Article)
                    WHERE a.main_content_embedding IS NOT NULL
                    RETURN COUNT(a) as count
                """)[0]['count']
                
                st.metric("Total Articles", article_count)
                st.metric("Articles with Embeddings", embedded_count)
            except Exception as e:
                st.error(f"Could not fetch statistics: {str(e)}")

    with tab2:
        embedding_experiment()
        
    with tab3:
        create_neo4j_node()

if __name__ == "__main__":
    main()
