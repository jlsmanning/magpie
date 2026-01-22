#FIXME: evaluate this placeholder code; does it makes sense with the input processing and search pipeline?
"""
Command-line interface for Magpie.

Interactive REPL for managing research interests and searching papers.
"""

import typing
from magpie.core.input_manager import process_message, ConversationResponse
from magpie.models.profile import UserProfile
from magpie.models.query import Query
from magpie.models.results import SearchResults


def load_profile() -> UserProfile:
    """Load user profile from disk."""
    # TODO: Implement profile loading from JSON file
    # For now, return empty profile
    return UserProfile(user_id="default_user")


def save_profile(profile: UserProfile) -> None:
    """Save user profile to disk."""
    # TODO: Implement profile saving to JSON file
    pass


def run_search_pipeline(query: Query, profile: UserProfile) -> SearchResults:
    """Execute the full search pipeline."""
    # TODO: Implement pipeline:
    # query → query_processor → vector_search → rerank → synthesize → results
    raise NotImplementedError("Search pipeline not yet implemented")


def display_results(results: SearchResults) -> None:
    """Display search results to user."""
    # TODO: Format and display papers nicely
    print(f"\nFound {len(results)} papers:")
    for i, result in enumerate(results.results, 1):
        paper = result.paper
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Published: {paper.published_date}")
        print(f"   Relevance: {result.relevance_score:.2f}")
        if result.explanation:
            print(f"   Why: {result.explanation}")


def apply_profile_updates(
    profile: UserProfile,
    updates: typing.Dict[str, typing.Any]
) -> UserProfile:
    """Apply updates to user profile."""
    # TODO: Parse updates dict and modify profile accordingly
    # Updates might be: {"add_interest": {...}}, {"remove_interest": "id"}, etc.
    return profile


def main():
    """Main REPL loop."""
    print("Welcome to Magpie")
    print("Your AI-powered research paper discovery assistant")
    print("Type 'exit' or 'quit' to exit\n")
    
    # Load user profile
    profile = load_profile()
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Process message
            response = process_message(user_input, profile, conversation_history)
            
            # Display conversational response
            print(f"\n{response.message_to_user}\n")
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response.message_to_user
            })
            
            # Handle structured actions
            if response.has_query():
                print("Searching for papers...")
                results = run_search_pipeline(response.query, profile)
                display_results(results)
                
                # Mark papers as seen
                for paper_id in results.get_paper_ids():
                    profile.mark_paper_seen(paper_id)
            
            if response.has_profile_updates():
                profile = apply_profile_updates(profile, response.profile_updates)
                save_profile(profile)
                print("✓ Profile updated")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
