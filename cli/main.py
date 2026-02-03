"""
Command-line interface for Magpie.

Interactive REPL for building queries and searching papers.
"""

import typing

from magpie.core.input_manager import process_message
from magpie.core.query_processor import process_query
from magpie.core.vector_search import VectorSearch
from magpie.core.reranker import rerank_results
from magpie.core.synthesizer import synthesize_results
from magpie.core.interactive_review import InteractiveReviewSession
from magpie.models.profile import UserProfile
from magpie.models.query import Query
from magpie.models.results import SearchResults
from magpie.integrations.embedder import Embedder
from magpie.integrations.llm_client import ClaudeClient
from magpie.utils.profile_manager import load_profile, save_profile
from magpie.utils.config import Config


def run_search_pipeline(query: Query, profile: UserProfile) -> SearchResults:
    """
    Execute the full search pipeline.
    
    Args:
        query: Query to execute
        profile: User profile
        
    Returns:
        SearchResults with ranked and synthesized papers
    """
    print("\n🔍 Processing query...")
    
    # Initialize components
    embedder = Embedder()
    vector_search = VectorSearch()
    llm_client = ClaudeClient()
    
    # Step 1: Process query (generate embeddings)
    print("  → Generating embeddings...")
    processed_query = process_query(query, embedder)
    
    # Step 2: Vector search
    print("  → Searching vector database...")
    results = vector_search.search(processed_query)
    print(f"  → Found {len(results.results)} papers")
    
    if not results.results:
        return results
    
    # Step 3: Rerank
    print("  → Reranking with LLM...")
    results = rerank_results(
        results,
        profile,
        llm_client=llm_client,
        rerank_top_n=20
    )
    
    # Step 4: Synthesize explanations
    print("  → Generating explanations...")
    results = synthesize_results(results, profile, llm_client=llm_client)
    
    # Mark papers as seen
    for paper_id in results.get_paper_ids():
        profile.mark_paper_seen(paper_id)
    
    print("✓ Search complete!\n")
    return results


def display_results(results: SearchResults) -> None:
    """Display search results to user."""
    if not results.results:
        print("\nNo papers found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results.results)} papers:")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results.results, 1):
        paper = result.paper
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        if len(paper.authors) > 3:
            print(f"            (+ {len(paper.authors) - 3} more)")
        print(f"   Published: {paper.published_date}")
        if paper.venue:
            print(f"   Venue: {paper.venue}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        
        if result.explanation:
            print(f"   {result.explanation}")
        
        print()


def interactive_review(results: SearchResults, profile: UserProfile) -> None:
    """
    Enter interactive review mode.
    
    Args:
        results: SearchResults to review
        profile: User profile
    """
    session = InteractiveReviewSession(results)
    
    # Start review
    print("\n" + "="*80)
    print("INTERACTIVE REVIEW")
    print("="*80 + "\n")
    print(session.start_review())
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            response = session.process_message(user_input)
            print(f"\n{response}")
            
            # Check if review is complete
            if "Review complete" in response or "Review ended" in response:
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting review...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main REPL loop."""
    print("=" * 80)
    print("Welcome to Magpie 🪶")
    print("Your AI-powered research paper discovery assistant")
    print("=" * 80)
    print("Type 'exit' or 'quit' to exit\n")
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set required environment variables in .env file")
        return
    
    # Load user profile
    print("Loading profile...")
    profile = load_profile()
    print(f"Loaded profile: {profile.user_id}\n")
    
    # Initialize conversation
    conversation_history = []
    llm_client = ClaudeClient()
    
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                save_profile(profile)
                break
            
            # Process message with InputManager
            response = process_message(
                user_message=user_input,
                profile=profile,
                conversation_history=conversation_history,
                llm_client=llm_client,
                auto_save=True  # Profile saves automatically
            )
            
            # Display conversational response
            print(f"\n{response.message_to_user}\n")
            
            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            conversation_history.append({
                "role": "assistant",
                "content": response.message_to_user
            })
            
            # Handle query execution
            if response.has_query():
                try:
                    # Run search pipeline
                    results = run_search_pipeline(response.query, profile)
                    
                    # Display results
                    display_results(results)
                    
                    # Save profile (papers marked as seen)
                    save_profile(profile)
                    
                    # Ask if user wants to review
                    if results.results:
                        print("Would you like to review these papers interactively? (yes/no)")
                        review_input = input("> ").strip().lower()
                        
                        if review_input in ['yes', 'y']:
                            interactive_review(results, profile)
                            # Save profile after review (in case papers were saved)
                            save_profile(profile)
                
                except Exception as e:
                    print(f"\nError during search: {e}")
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            save_profile(profile)
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
