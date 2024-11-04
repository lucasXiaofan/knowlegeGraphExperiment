from openai import OpenAI
import os

def rewrite_query(original_query: str, client: OpenAI) -> str:
    """
    Rewrites the search query to be more focused and intention-based.
    
    Args:
        original_query (str): The original user query
        client (OpenAI): OpenAI client instance
    
    Returns:
        str: Rewritten, focused query
    """
    prompt = """Extract the core technical intention or question from this query. 
    Make it concise and focused on the technical requirement.
    Remove any conversational elements or unnecessary context.
    
    Original query: {query}
    
    Respond with only the rewritten query, nothing else."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a technical query optimizer. Convert verbose queries into precise technical requirements."},
                {"role": "user", "content": prompt.format(query=original_query)}
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return original_query  # Return original query if rewrite fails