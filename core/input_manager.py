#FIXME: is this the most efficient structure for doing this?
"""
Input manager for Magpie.

Handles conversational interaction with user, parsing intents and generating
structured outputs (Query objects, profile updates) from natural language.
"""

import typing
from magpie.models.query import Query
from magpie.models.profile import UserProfile


class ConversationResponse:
    """
    Response from processing user input.
    
    Contains conversational message plus any structured actions to take.
    """
    def __init__(
        self,
        message_to_user: str,
        query: typing.Optional[Query] = None,
        profile_updates: typing.Optional[typing.Dict[str, typing.Any]] = None
    ):
        self.message_to_user = message_to_user
        self.query = query
        self.profile_updates = profile_updates
    
    def has_query(self) -> bool:
        """Check if response includes a query to execute."""
        return self.query is not None
    
    def has_profile_updates(self) -> bool:
        """Check if response includes profile changes to apply."""
        return self.profile_updates is not None


def process_message(
    user_message: str,
    profile: UserProfile,
    conversation_history: typing.List[typing.Dict[str, str]]
) -> ConversationResponse:
    """
    Process user message in conversational context.
    
    LLM handles the conversation naturally and returns structured outputs
    when actions are needed (searching papers or updating profile).
    
    Args:
        user_message: Latest user input
        profile: Current user profile with interests and preferences
        conversation_history: Previous messages in format:
            [{"role": "user"|"assistant", "content": "..."}]
    
    Returns:
        ConversationResponse containing:
        - message_to_user: Conversational response to display
        - query: Query object if user wants papers (triggers search pipeline)
        - profile_updates: Dict of profile changes if user wants to modify interests
    
    Example conversation flows:
        User: "I'm interested in explainable AI"
        → Returns profile_updates to add interest
        
        User: "Find me some papers"
        → Returns Query based on profile interests
        
        User: "I care more about XAI than CV"
        → Returns profile_updates to adjust star ratings
        
        User: "What are my interests?"
        → Returns just conversational message listing interests
    """
    
    # TODO: Implement LLM call with prompt structure:
    # 
    # System prompt:
    # - You help manage research interests and find papers
    # - When user wants papers, generate Query JSON
    # - When user modifies interests, generate profile update JSON
    # - Otherwise just respond conversationally
    #
    # Context:
    # - Current profile interests: {profile.interests}
    # - Conversation history: {conversation_history}
    # - User message: {user_message}
    #
    # Response format:
    # - Conversational text
    # - Optional JSON for actions:
    #   {"action": "query", "queries": [...], "params": {...}}
    #   {"action": "update_profile", "changes": {...}}
    
    # Stub: Return simple response
    return ConversationResponse(
        message_to_user="[Stub] I understand. LLM integration coming soon.",
        query=None,
        profile_updates=None
    )
