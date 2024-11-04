from langchain_community.graphs import Neo4jGraph
import json
# from dotenv import load_dotenv
import os
import warnings
import streamlit as st
from openai import OpenAI
import uuid
from datetime import datetime

embedding_client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"])
kg = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],     
    username=st.secrets["NEO4J_USERNAME"], 
    password=st.secrets["NEO4J_PASSWORD"] 
)

# Create merge query for articles
merge_article_node_query = """
MERGE(article:Article {aid: $articleParam.aid})
    ON CREATE SET 
        article.title = $articleParam.article_title,
        article.modify_date = $articleParam.modify_date,
        article.main_content = $articleParam.main_content,
        article.use_case = $articleParam.use_case,
        article.url = $articleParam.url
RETURN article
"""

def setup_neo4j_database(neo4j_uri, neo4j_username, neo4j_password):
    """Setup Neo4j database with constraints and load articles"""
    
    # Initialize Neo4j connection
    kg = Neo4jGraph(
        url=neo4j_uri, 
        username=neo4j_username, 
        password=neo4j_password, 
    )
    
    # Create unique constraint for article ID
    kg.query("""
    CREATE CONSTRAINT unique_article IF NOT EXISTS 
        FOR (a:Article) REQUIRE a.aid IS UNIQUE
    """)
    
    # Load articles from JSON
    with open('final_articles_with_ids.json', 'r') as f:
        articles = json.load(f)
    
    # Insert articles into Neo4j
    for article in articles:
        kg.query(
            merge_article_node_query,
            {'articleParam': article}
        )
    
    print(f"Loaded {len(articles)} articles into Neo4j database")
    return kg

def neo4j_vector_search(question, use_case=False, top_k=1, openai_api_key=None):
    """
    Search for similar articles using Neo4j vector index
    
    Args:
        question (str): The search query
        use_case (bool): Whether to search for use_case or main_content
        top_k (int): Number of results to return
    """
    index_name = "article_content_embedding" if not use_case else "article_usecase_embedding"
    field_name = "main_content" if not use_case else "use_case"
    
    vector_search_query = """
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
            token: $openAiApiKey
        }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) 
    yield node, score
    RETURN 
        score, 
        node.title AS title,
        node.url AS url
    """
    
    similar = kg.query(
        vector_search_query, 
        params={
            'question': question,
            'openAiApiKey': openai_api_key,
            'index_name': index_name,
            'top_k': top_k
        }
    )
    
    return similar

def create_neo4j_node():
    st.title("Create Neo4j Node")
    
    # Text input for knowledge
    knowledge_text = st.text_area("Enter knowledge text:", height=150)
    
    if st.button("Save to Neo4j"):
        if knowledge_text:
            try:
                with st.spinner("Processing..."):
                    # Generate knowledge ID
                    knowledge_id = str(uuid.uuid4())
                    
                    # Get text embedding
                    response = embedding_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=knowledge_text
                    )
                    embedding = response.data[0].embedding
                    
                    # Create Neo4j node
                    create_node_query = """
                    CREATE (n:macInCloudService {
                        knowledge_id: $knowledge_id,
                        knowledge_text: $knowledge_text,
                        creation_date: datetime($creation_date),
                        text_embedding: $embedding
                    })
                    RETURN n
                    """
                    
                    # Execute query
                    result = kg.query(
                        create_node_query,
                        params={
                            'knowledge_id': knowledge_id,
                            'knowledge_text': knowledge_text,
                            'creation_date': datetime.now().isoformat(),
                            'embedding': embedding
                        }
                    )
                    
                    st.success(f"Node created successfully with ID: {knowledge_id}")
                    
                    # Show created node details
                    st.json({
                        'knowledge_id': knowledge_id,
                        'knowledge_text': knowledge_text,
                        'creation_date': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.error(f"Error creating node: {str(e)}")
        else:
            st.warning("Please enter knowledge text")
    
    # Display existing nodes
    if st.button("View Existing Nodes"):
        with st.spinner("Fetching nodes..."):
            try:
                query = """
                MATCH (n:macInCloudService)
                RETURN n.knowledge_id, n.knowledge_text, n.creation_date
                ORDER BY n.creation_date DESC
                LIMIT 10
                """
                results = kg.query(query)
                
                if results:
                    st.subheader("Recent Nodes (Last 10)")
                    for row in results:
                        with st.expander(f"Node: {row['n.knowledge_id']}"):
                            st.write("**Knowledge Text:**")
                            st.write(row['n.knowledge_text'])
                            st.write("**Created:**", row['n.creation_date'])
                            
                            # Add delete button for each node
                            if st.button(f"Delete Node", key=f"delete_{row['n.knowledge_id']}"):
                                try:
                                    delete_query = """
                                    MATCH (n:macInCloudService {knowledge_id: $knowledge_id})
                                    DELETE n
                                    """
                                    kg.query(delete_query, params={'knowledge_id': row['n.knowledge_id']})
                                    st.success(f"Node {row['n.knowledge_id']} deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting node: {str(e)}")
                else:
                    st.info("No nodes found")
                    
            except Exception as e:
                st.error(f"Error fetching nodes: {str(e)}")

def mac_service_vector_search(question_embedding, top_k=1):
    """
    Search for similar knowledge in macInCloudService nodes using pre-calculated embedding
    """
    vector_search_query = """
    CALL db.index.vector.queryNodes("mac_service_embedding", $top_k, $question_embedding) 
    yield node, score
    RETURN 
        score, 
        'knowledge_base' as source,
        'Knowledge Base Entry' as title,
        node.knowledge_text as content,
        node.knowledge_id as id,
        node.creation_date as date
    ORDER BY score DESC
    """
    
    return kg.query(
        vector_search_query, 
        params={
            'question_embedding': question_embedding,
            'top_k': top_k
        }
    )

def article_vector_search(question_embedding, top_k=1):
    """
    Search for similar content in articles using pre-calculated embedding
    """
    vector_search_query = """
    CALL db.index.vector.queryNodes("article_content_embedding", $top_k, $question_embedding) 
    yield node, score
    RETURN 
        score,
        'article' as source,
        node.title as title,
        node.main_content as content,
        node.url as url
    ORDER BY score DESC
    """
    
    return kg.query(
        vector_search_query,
        params={
            'question_embedding': question_embedding,
            'top_k': top_k
        }
    )

def combined_vector_search(question, top_k=5):
    """
    Combine results from both article and macInCloudService searches
    """
    # Calculate embedding once
    response = embedding_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_embedding = response.data[0].embedding
    
    # Get results from both sources using the same embedding
    article_results = article_vector_search(
        question_embedding=question_embedding,
        top_k=top_k
    )
    
    mac_results = mac_service_vector_search(
        question_embedding=question_embedding,
        top_k=top_k
    )
    
    # Format article results
    formatted_articles = [{
        'score': result['score'],
        'source': 'article',
        'title': result['title'],
        'content': result.get('content', ''),
        'url': result.get('url', ''),
        'knowledge_id': None,
        'creation_date': None
    } for result in article_results]
    
    # Format mac service results
    formatted_mac = [{
        'score': result['score'],
        'source': 'knowledge_base',
        'title': result['title'],
        'content': result['content'],
        'url': None,
        'knowledge_id': result['id'],
        'creation_date': result['date']
    } for result in mac_results]
    
    # Combine and sort all results
    all_results = formatted_articles + formatted_mac
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top_k results
    return all_results

if __name__ == "__main__":
    pass
    # Load environment variables for Neo4j connection
    # load_dotenv()

    # kg = Neo4jGraph(
    #     url=NEO4J_URI, 
    #     username=NEO4J_USERNAME, 
    #     password=NEO4J_PASSWORD, 
    # )
    # kg.query("""
    # DROP INDEX article_content_embedding IF EXISTS
    # """)

    # kg.query("""
    # DROP INDEX article_usecase_embedding IF EXISTS
    # """)
    # kg.query("""
    # CREATE VECTOR INDEX article_content_embedding IF NOT EXISTS
    # FOR (a:Article) ON (a.main_content_embedding)  // Changed from main_content to main_content_embedding
    # OPTIONS { indexConfig: {
    #     `vector.dimensions`: 1536,
    #     `vector.similarity_function`: 'cosine'
    # }}
    # """)

    # kg.query("""
    # CREATE VECTOR INDEX article_usecase_embedding IF NOT EXISTS
    # FOR (a:Article) ON (a.use_case_embedding)  // Changed from use_case to use_case_embedding
    # OPTIONS { indexConfig: {
    #     `vector.dimensions`: 1536,
    #     `vector.similarity_function`: 'cosine'
    # }}
    # """)
    # vector_search_result = neo4j_vector_search(question="Hello. We installed some Flutter libraries. We request that you please set the default version of XCODE 15.4, at this time the default is 16 and it is causing us problems. Thank you.",
    #                                             use_case=False, 
    #                                             top_k=5, 
    #                                             openai_api_key=OPENAI_API_KEY)
    # print(f"Vector search result: {vector_search_result}")


    
    # print(kg.query("""
    # MATCH (a:Article) where a.aid = "version0_article_0069"
    # RETURN a.main_content_embedding
    # """))
    # print(kg.query("SHOW INDEXES")) 
  
    # # Setup database and load articles
    # # kg = setup_neo4j_database(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) 

    # # Add vector embeddings for main_content
    # kg.query("""
    #     MATCH (a:Article) WHERE a.main_content_embedding IS NULL
    #     WITH a, genai.vector.encode(
    #         a.main_content,
    #         "OpenAI",
    #         {
    #             token: $openAiApiKey
    #         }) AS vector
    #     CALL db.create.setNodeVectorProperty(a, "main_content_embedding", vector)
    #     """,
    #     params={"openAiApiKey": OPENAI_API_KEY})

    # # Add vector embeddings for use_case
    # kg.query("""
    #     MATCH (a:Article) WHERE a.use_case_embedding IS NULL
    #     WITH a, genai.vector.encode(
    #         a.use_case,
    #         "OpenAI",
    #         {
    #             token: $openAiApiKey
    #         }) AS vector
    #     CALL db.create.setNodeVectorProperty(a, "use_case_embedding", vector)
    #     """,
    #     params={"openAiApiKey": OPENAI_API_KEY})
    # print(kg.schema)
    # kg.refresh_schema()
    # print(kg.schema)