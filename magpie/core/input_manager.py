"""
Input manager for Magpie.

Handles conversational interaction with user, parsing intents and generating
structured outputs (Query objects, profile updates) from natural language.
Uses LLM to help users build search queries through dialogue.
"""

import datetime
import typing

from magpie.models.query import Query, SubQuery
from magpie.models.profile import UserProfile
from magpie.integrations.llm_client import ClaudeClient


# Conversational system prompt for query building
CONVERSATIONAL_QUERY_BUILDER_PROMPT = """
You are a helpful research assistant helping a user build a search query for finding academic papers.

YOUR ROLE:
- Help users discover what they want to search for through conversation
- Suggest relevant search topics based on papers they mention or interests they describe
- Clarify ambiguous requests before taking action
- Explain what you're doing in plain language
- Be genuinely helpful, not just a command parser

USER CONTEXT:
{user_context}

CURRENT QUERY STATE:
{current_query_summary}

CONVERSATION HISTORY:
You have access to the full conversation history to understand context.

RESPONSE TYPES:

1. CONVERSATION (just talking, no action):
Use when: Answering questions, explaining, discussing, helping user think through what to search
{{
  "response_type": "conversation",
  "message": "Your natural response here"
}}

2. QUESTION (asking for clarification):
Use when: User request is ambiguous, or you want to suggest options
{{
  "response_type": "question",
  "message": "Would you like to search for X, Y, or Z?",
  "suggestions": ["option1", "option2", "option3"]
}}

3. ACTION (modifying the query):
Use when: User clearly wants to add/remove/modify query, or has confirmed a suggestion
{{
  "response_type": "action",
  "message": "I've added 'topic' to your query.",
  "action": {{
    "type": "<action_type>",
    // ... parameters
  }}
}}

ACTION TYPES:

Basic query modification:
- {{"type": "add_subquery", "text": "search topic"}}
- {{"type": "add_multiple_subqueries", "texts": ["topic1", "topic2"]}}
- {{"type": "remove_subquery", "target": "topic to remove"}}
- {{"type": "clear_query"}}
- {{"type": "modify_subquery_text", "target": "old topic", "new_text": "new topic"}}

Weight management:
- {{"type": "set_weights", "weights": {{"topic1": 0.6, "topic2": 0.4}}}}
- {{"type": "increase_weight", "target": "topic"}}
- {{"type": "decrease_weight", "target": "topic"}}

Query parameters:
- {{"type": "set_max_results", "value": 20}}
- {{"type": "set_date_range", "start_date": "2023-01-01", "end_date": "2025-01-01"}}
- {{"type": "set_min_citations", "value": 50}}
- {{"type": "set_venues", "venues": ["CVPR", "NeurIPS"]}}
- {{"type": "set_recency_weight", "value": 0.7}}

Context management:
- {{"type": "update_context", "addition": "text to add to context", "requires_confirmation": true}}
- {{"type": "set_context", "new_context": "complete new context text"}}
- {{"type": "clear_context"}}
- {{"type": "show_context"}}

Viewing and execution:
- {{"type": "show_query"}}
- {{"type": "execute"}}

EXAMPLE INTERACTIONS:

User: "I really like this paper [pastes abstract about transformers]"
You: {{
  "response_type": "question",
  "message": "This paper is about transformer architectures for sequence modeling. Would you like to search for papers on: transformer architectures, attention mechanisms, or neural machine translation?",
  "suggestions": ["transformer architectures", "attention mechanisms", "neural machine translation"]
}}

User: "Yes, transformers and attention"
You: {{
  "response_type": "action",
  "message": "I've added 'transformer architectures' and 'attention mechanisms' to your query. Both are weighted equally at 50%. Would you like to adjust the weights or add more topics?",
  "action": {{"type": "add_multiple_subqueries", "texts": ["transformer architectures", "attention mechanisms"]}}
}}

User: "Make transformers more important"
You: {{
  "response_type": "action",
  "message": "I've increased the weight for 'transformer architectures'. It's now 60% and 'attention mechanisms' is 40%.",
  "action": {{"type": "increase_weight", "target": "transformer architectures"}}
}}

User: "What's in my query right now?"
You: {{
  "response_type": "action",
  "message": "Your current query searches for:\\n1. Transformer architectures (60%)\\n2. Attention mechanisms (40%)\\n\\nWant to modify anything or ready to search?",
  "action": {{"type": "show_query"}}
}}

User: "Run it"
You: {{
  "response_type": "action",
  "message": "Searching for papers now...",
  "action": {{"type": "execute"}}
}}

User: "I'm not sure what to search for"
You: {{
  "response_type": "conversation",
  "message": "I can help! Are you exploring a new research area, or do you have a specific problem or paper in mind that you'd like to find related work for?"
}}

User: "Update my context: I'm focusing on multimodal learning now"
You: {{
  "response_type": "action",
  "message": "I've updated your research context to note your focus on multimodal learning. This will help me suggest more relevant searches.",
  "action": {{"type": "set_context", "new_context": "Focusing on multimodal learning"}}
}}

User: "Show me my research context"
You: {{
  "response_type": "action",
  "message": "Your research context: [displays current context]. Would you like to update or modify this?",
  "action": {{"type": "show_context"}}
}}

GUIDELINES:
- Be conversational and natural, not robotic
- When user mentions a paper, analyze it and suggest relevant search topics
- Ask clarifying questions for ambiguous requests
- Confirm understanding before taking irreversible actions (like clearing query)
- Use "conversation" type liberally - not every message needs an action
- Use "question" type when suggesting options
- Only use "action" type when you're confident about what to do
- If you learn something significant about the user's research goals or preferences during conversation, proactively suggest updating their context with update_context action
- If query is empty and user says "search", return: {{"response_type": "action", "message": "You don't have any topics in your query yet. What would you like to search for?", "action": {{"type": "error"}}}}
- Be helpful in explaining what searches will do

Remember: You're a collaborative assistant, not just a command parser. Help users discover what they want to search for!
"""


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
    conversation_history: typing.List[typing.Dict[str, str]],
    llm_client: typing.Optional[ClaudeClient] = None,
    auto_save: bool = True    
) -> ConversationResponse:
    """
    Process user message in conversational context.
    
    LLM helps user build query through natural dialogue, suggesting topics,
    clarifying ambiguities, and taking actions when appropriate.
    
    Args:
        user_message: Latest user input
        profile: Current user profile with current_query
        conversation_history: Previous messages in format:
            [{"role": "user"|"assistant", "content": "..."}]
        llm_client: LLM client for conversation (creates if None)
        auto_save: If True, automatically saves profile when modified (default: True)
    
    Returns:
        ConversationResponse containing:
        - message_to_user: Conversational response to display
        - query: Query object if user wants to execute search
        - profile_updates: Dict indicating profile was modified
    """
    if llm_client is None:
        llm_client = ClaudeClient()
    
    # Format current query state for LLM
    query_summary = _format_current_query(profile.current_query)
    
    # Format user context
    user_context = profile.research_context or "No research context provided yet."
    
    # Build system prompt with current context
    system_prompt = CONVERSATIONAL_QUERY_BUILDER_PROMPT.format(
        user_context=user_context,
        current_query_summary=query_summary
    )
    
    # Add user message to conversation
    messages = conversation_history + [
        {"role": "user", "content": user_message}
    ]
    
    # Get LLM response
    try:
        llm_response = llm_client.chat_with_json(
            messages=messages,
            system=system_prompt,
            max_tokens=2048,
            temperature=0.3  # Slightly creative but mostly deterministic
        )
    except Exception as e:
        return ConversationResponse(
            message_to_user=f"Sorry, I encountered an error: {e}",
            query=None,
            profile_updates=None
        )
    
    # Handle response based on type
    response_type = llm_response.get("response_type", "conversation")
    message = llm_response.get("message", "")
    
    if response_type == "action":
        action = llm_response.get("action", {})
        return _handle_action(action, message, profile)

    # Auto-save profile if it was modified
        if auto_save and response.has_profile_updates():
            from magpie.utils.profile_manager import save_profile
            try:
                save_profile(profile)
            except Exception as e:
                print(f"Warning: Failed to save profile: {e}")
        
        return response
    else:
        # conversation or question - just return message
        return ConversationResponse(
            message_to_user=message,
            query=None,
            profile_updates=None
        )


def _format_current_query(current_query: typing.Optional[Query]) -> str:
    """
    Format current query state for display to LLM.
    
    Args:
        current_query: Current query being built, or None
        
    Returns:
        Human-readable summary of query state
    """
    if current_query is None:
        return "No query has been created yet."
    
    if not current_query.queries:
        return "Query exists but has no topics yet."
    
    lines = ["Current query contains:"]
    for i, sq in enumerate(current_query.queries, 1):
        lines.append(f"  {i}. '{sq.text}' (weight: {sq.weight:.0%})")
    
    # Add parameters if set
    if current_query.max_results != 10:
        lines.append(f"Max results: {current_query.max_results}")
    if current_query.date_range:
        lines.append(f"Date range: {current_query.date_range[0]} to {current_query.date_range[1]}")
    if current_query.min_citations:
        lines.append(f"Minimum citations: {current_query.min_citations}")
    if current_query.venues:
        lines.append(f"Preferred venues: {', '.join(current_query.venues)}")
    
    return "\n".join(lines)


def _handle_action(
    action: typing.Dict[str, typing.Any],
    message: str,
    profile: UserProfile
) -> ConversationResponse:
    """
    Execute action on profile and return appropriate response.
    
    Args:
        action: Action dict from LLM with type and parameters
        message: Message to show user
        profile: User profile to modify
        
    Returns:
        ConversationResponse with results of action
    """
    action_type = action.get("type")
    
    try:
        # Execute action
        if action_type == "add_subquery":
            _add_subquery(profile, action["text"])
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "add_multiple_subqueries":
            for text in action["texts"]:
                _add_subquery(profile, text, rebalance=False)
            _rebalance_weights(profile.current_query)
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "remove_subquery":
            _remove_subquery(profile, action["target"])
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "clear_query":
            profile.current_query = None
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "set_weights":
            _set_weights(profile, action["weights"])
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "increase_weight":
            _adjust_weight(profile, action["target"], increase=True)
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "decrease_weight":
            _adjust_weight(profile, action["target"], increase=False)
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "set_max_results":
            if profile.current_query:
                profile.current_query.max_results = action["value"]
                profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "set_date_range":
            if profile.current_query:
                start = datetime.datetime.strptime(action["start_date"], "%Y-%m-%d").date() if action.get("start_date") else None
                end = datetime.datetime.strptime(action["end_date"], "%Y-%m-%d").date() if action.get("end_date") else None
                profile.current_query.date_range = (start, end) if start and end else None
                profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "update_context":
            # LLM suggests adding to context
            addition = action.get("addition", "")
            if profile.research_context:
                profile.research_context += f" {addition}"
            else:
                profile.research_context = addition
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "set_context":
            # User explicitly sets entire context
            profile.research_context = action.get("new_context", "")
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "clear_context":
            profile.research_context = None
            profile.last_updated = datetime.datetime.now()
            return ConversationResponse(message, query=None, profile_updates={"modified": True})
        
        elif action_type == "show_context":
            # LLM has already formatted context in message
            return ConversationResponse(message, query=None, profile_updates=None)
        
        elif action_type == "show_query":
            # Just return message (LLM has already formatted the query info)
            return ConversationResponse(message, query=None, profile_updates=None)
        
        elif action_type == "execute":
            if profile.current_query is None or not profile.current_query.queries:
                return ConversationResponse(
                    message_to_user="You don't have any topics in your query yet. What would you like to search for?",
                    query=None,
                    profile_updates=None
                )
            # Finalize query with profile defaults and return for execution
            query = _finalize_query(profile.current_query, profile)
            return ConversationResponse(message, query=query, profile_updates=None)
        
        elif action_type == "error":
            # LLM flagged an error condition
            return ConversationResponse(message, query=None, profile_updates=None)
        
        else:
            return ConversationResponse(
                message_to_user=f"Unknown action type: {action_type}",
                query=None,
                profile_updates=None
            )
    
    except Exception as e:
        return ConversationResponse(
            message_to_user=f"Error executing action: {e}",
            query=None,
            profile_updates=None
        )


def _add_subquery(profile: UserProfile, text: str, rebalance: bool = True) -> None:
    """Add subquery to current_query, creating query if needed."""
    # Create query if doesn't exist
    if profile.current_query is None:
        profile.current_query = Query(
            queries=[],
            max_results=profile.max_results,
            date_range=profile.date_range,
            min_citations=profile.min_citations,
            recency_weight=profile.recency_weight,
            venues=profile.venues,
            exclude_seen_papers=profile.exclude_seen_papers
        )
    
    # Add subquery
    subquery = SubQuery(
        text=text,
        weight=1.0,  # Will be rebalanced
        source_interest_ids=None
    )
    profile.current_query.queries.append(subquery)
    
    # Rebalance weights
    if rebalance:
        _rebalance_weights(profile.current_query)


def _remove_subquery(profile: UserProfile, target: str) -> None:
    """Remove subquery matching target text."""
    if profile.current_query is None:
        return
    
    # Find and remove matching subquery (case-insensitive)
    target_lower = target.lower()
    profile.current_query.queries = [
        sq for sq in profile.current_query.queries
        if target_lower not in sq.text.lower()
    ]
    
    # Rebalance remaining
    if profile.current_query.queries:
        _rebalance_weights(profile.current_query)


def _rebalance_weights(query: Query) -> None:
    """Rebalance all subquery weights to sum to 1.0."""
    if not query.queries:
        return
    
    n = len(query.queries)
    for sq in query.queries:
        sq.weight = 1.0 / n


def _set_weights(profile: UserProfile, weights: typing.Dict[str, float]) -> None:
    """Set explicit weights for subqueries."""
    if profile.current_query is None:
        return
    
    # Match weights to subqueries by text (case-insensitive)
    for sq in profile.current_query.queries:
        for topic, weight in weights.items():
            if topic.lower() in sq.text.lower():
                sq.weight = weight
                break
    
    # Normalize to sum to 1.0
    total = sum(sq.weight for sq in profile.current_query.queries)
    if total > 0:
        for sq in profile.current_query.queries:
            sq.weight /= total


def _adjust_weight(profile: UserProfile, target: str, increase: bool) -> None:
    """Increase or decrease weight for target subquery."""
    if profile.current_query is None or not profile.current_query.queries:
        return
    
    # Find target subquery
    target_lower = target.lower()
    target_sq = None
    for sq in profile.current_query.queries:
        if target_lower in sq.text.lower():
            target_sq = sq
            break
    
    if target_sq is None:
        return
    
    # Adjust by ±0.1
    adjustment = 0.1 if increase else -0.1
    target_sq.weight += adjustment
    
    # Clamp to reasonable range
    target_sq.weight = max(0.1, min(0.9, target_sq.weight))
    
    # Normalize all weights to sum to 1.0
    total = sum(sq.weight for sq in profile.current_query.queries)
    for sq in profile.current_query.queries:
        sq.weight /= total


def _finalize_query(current_query: Query, profile: UserProfile) -> Query:
    """
    Finalize query before execution, filling in any missing defaults.
    
    Args:
        current_query: Query being finalized
        profile: User profile with defaults
        
    Returns:
        Query ready for execution
    """
    # Query is already built, just ensure it has all needed fields
    # (They were copied from profile when query was created)
    return current_query
