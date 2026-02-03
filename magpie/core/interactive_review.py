"""
Interactive review for Magpie.

Post-search conversation where user explores papers one at a time,
asks questions, and decides whether to save or skip each paper.
"""

import typing

from magpie.models.results import SearchResults
from magpie.models.paper import Paper
from magpie.integrations.llm_client import ClaudeClient
from magpie.integrations.pdf_fetcher import PDFFetcher
from magpie.integrations.zotero_client import ZoteroClient
from magpie.utils.config import Config


class InteractiveReviewSession:
    """
    Manages interactive review of search results.
    
    Presents papers one at a time, allows user to ask questions
    (potentially downloading PDFs to answer), and save to Zotero or skip.
    """
    
    def __init__(
        self,
        results: SearchResults,
        llm_client: typing.Optional[ClaudeClient] = None,
        pdf_fetcher: typing.Optional[PDFFetcher] = None,
        zotero_client: typing.Optional[ZoteroClient] = None
    ):
        """
        Initialize review session.
        
        Args:
            results: SearchResults from search pipeline
            llm_client: LLM client for conversation (creates if None)
            pdf_fetcher: PDF fetcher for downloading papers (creates if None)
            zotero_client: Zotero client for saving papers (creates if None and configured)
        """
        self.results = results
        self.current_index = 0
        self.downloaded_pdfs: typing.Dict[str, typing.Tuple[str, str]] = {}  # paper_id -> (path, text)
        self.saved_to_zotero: typing.Set[str] = set()
        self.conversation_history: typing.List[typing.Dict[str, str]] = []
        self.last_response: str = ""
        
        # Initialize clients
        self.llm_client = llm_client or ClaudeClient()
        self.pdf_fetcher = pdf_fetcher or PDFFetcher()
        
        # Zotero is optional
        self.zotero_client = zotero_client
        if self.zotero_client is None and Config.ZOTERO_LIBRARY_ID and Config.ZOTERO_API_KEY:
            try:
                self.zotero_client = ZoteroClient()
            except Exception as e:
                print(f"Note: Zotero not available: {e}")
    
    def start_review(self) -> str:
        """
        Start the review session.
        
        Returns:
            Initial message presenting first paper
        """
        if not self.results.results:
            return "No papers to review."
        
        message = f"I found {len(self.results.results)} papers. Ready to review them one by one?"
        self.last_response = message
        return message
    
    def process_message(self, user_message: str) -> str:
        """
        Process user message during review.
        
        Handles questions about current paper, commands to save/skip/navigate,
        and references to other papers.
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build prompt with current context
        system_prompt = self._build_review_prompt()
        
        # Get LLM response
        try:
            llm_response = self.llm_client.chat_with_json(
                messages=self.conversation_history,
                system=system_prompt,
                max_tokens=2048,
                temperature=0.3
            )
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"
            self._add_to_history(response)
            return response
        
        # Handle response based on action
        action_type = llm_response.get("action")
        
        if action_type == "present_paper":
            response = self._present_current_paper()
        
        elif action_type == "discuss_current":
            question = llm_response.get("question", user_message)
            needs_pdf = llm_response.get("needs_pdf", False)
            response = self._discuss_paper(question, needs_pdf)
        
        elif action_type == "save_current":
            response = self._save_current_paper()
        
        elif action_type == "skip_current":
            response = self._skip_current_paper()
        
        elif action_type == "switch_to":
            paper_index = llm_response.get("paper_index")
            if paper_index is not None:
                response = self._switch_to_paper(paper_index)
            else:
                response = "I couldn't determine which paper you wanted to switch to. Please specify a paper number."
        
        elif action_type == "exit_review":
            response = self._exit_review()
        
        else:
            # Conversational response without action
            response = llm_response.get("message", "I'm not sure what you'd like me to do.")
        
        self._add_to_history(response)
        return response
    
    def _build_review_prompt(self) -> str:
        """Build system prompt for review conversation."""
        current_paper = self._get_current_paper()
        paper_id_str = f"{current_paper.paper_id[0]}:{current_paper.paper_id[1]}"
        
        has_pdf = paper_id_str in self.downloaded_pdfs
        zotero_available = self.zotero_client is not None
        
        prompt = f"""You are helping a user review research papers one at a time.

CURRENT PAPER ({self.current_index + 1} of {len(self.results.results)}):
Title: {current_paper.title}
Authors: {', '.join(current_paper.authors[:3])}
Published: {current_paper.published_date}
PDF available: {has_pdf}

CONTEXT:
- User is reviewing papers from a search
- You present each paper's explanation
- User can ask questions (you may need to read the PDF to answer)
- User can save to Zotero or skip to next paper
- User can reference other papers by number, title, or description

ACTIONS YOU CAN TAKE:

{{"action": "present_paper", "message": "..."}}
Use when user is ready to hear about the current paper. Read the synthesis explanation.

{{"action": "discuss_current", "question": "...", "needs_pdf": true/false, "message": "..."}}
Use when user asks a question about current paper. Set needs_pdf=true if you need to read the full paper to answer.

{{"action": "save_current", "message": "Saving to Zotero..."}}
Use when user wants to save current paper.

{{"action": "skip_current", "message": "Moving to next paper..."}}
Use when user wants to skip current paper.

{{"action": "switch_to", "paper_index": N, "message": "..."}}
Use when user references a different paper. Index is 0-based.

{{"action": "exit_review", "message": "..."}}
Use when user wants to stop reviewing.

{{"message": "..."}}
Use for general conversation without specific action.

GUIDELINES:
- Be conversational and helpful
- For voice interface: keep responses concise (2-4 sentences)
- If user's question requires PDF and you don't have it, set needs_pdf=true
- Zotero available: {zotero_available}
"""
        
        return prompt
    
    def _present_current_paper(self) -> str:
        """Present current paper's synthesis explanation."""
        paper_result = self.results.results[self.current_index]
        paper = paper_result.paper
        
        response = f"Paper {self.current_index + 1} of {len(self.results.results)}: {paper.title}\n\n"
        
        if paper_result.explanation:
            response += paper_result.explanation
        else:
            response += f"This paper by {', '.join(paper.authors[:2])} was published in {paper.published_date.year}."
        
        response += "\n\nWould you like to know more, skip it, or save to Zotero?"
        
        return response
    
    def _discuss_paper(self, question: str, needs_pdf: bool) -> str:
        """
        Discuss current paper, potentially reading PDF to answer question.
        
        Args:
            question: User's question
            needs_pdf: Whether PDF is needed to answer
            
        Returns:
            Answer to question
        """
        paper = self._get_current_paper()
        paper_id_str = f"{paper.paper_id[0]}:{paper.paper_id[1]}"
        
        # Get PDF if needed
        pdf_text = None
        if needs_pdf:
            if paper_id_str in self.downloaded_pdfs:
                # Already have PDF
                _, pdf_text = self.downloaded_pdfs[paper_id_str]
            elif paper.pdf_url:
                # Download PDF
                try:
                    pdf_path, pdf_text = self.pdf_fetcher.fetch_and_extract(
                        paper.pdf_url,
                        paper_id_str
                    )
                    self.downloaded_pdfs[paper_id_str] = (pdf_path, pdf_text)
                except Exception as e:
                    return f"I couldn't download the PDF to answer that: {e}. I can only tell you about the abstract."
            else:
                return "This paper doesn't have a PDF available. I can only answer based on the abstract."
        
        # Build context for LLM
        context = f"Paper: {paper.title}\nAbstract: {paper.abstract}\n"
        if pdf_text:
            # Limit PDF text to avoid token limits (first ~3000 words)
            words = pdf_text.split()[:3000]
            context += f"\nFull paper text:\n{' '.join(words)}...\n"
        
        # Ask LLM to answer question
        answer_prompt = f"{context}\nUser question: {question}\n\nAnswer concisely (2-3 sentences):"
        
        try:
            answer = self.llm_client.chat(
                messages=[{"role": "user", "content": answer_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return answer
        except Exception as e:
            return f"Sorry, I couldn't answer that: {e}"
    
    def _save_current_paper(self) -> str:
        """Save current paper to Zotero."""
        if self.zotero_client is None:
            return "Zotero is not configured. Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY in .env to save papers."
        
        paper = self._get_current_paper()
        paper_id_str = f"{paper.paper_id[0]}:{paper.paper_id[1]}"
        
        # Check if already saved
        if paper_id_str in self.saved_to_zotero:
            return "This paper is already saved to Zotero. Moving to next..."
        
        # Get PDF path if downloaded
        pdf_path = None
        if paper_id_str in self.downloaded_pdfs:
            pdf_path, _ = self.downloaded_pdfs[paper_id_str]
        elif paper.pdf_url:
            # Download for Zotero
            try:
                pdf_path, pdf_text = self.pdf_fetcher.fetch_and_extract(
                    paper.pdf_url,
                    paper_id_str
                )
                self.downloaded_pdfs[paper_id_str] = (pdf_path, pdf_text)
            except Exception:
                pass  # Save without PDF if download fails
        
        # Save to Zotero
        try:
            # Add tags from matched queries
            tags = [sq.text for sq in self.results.results[self.current_index].matched_subqueries]
            
            self.zotero_client.save_paper(
                paper,
                tags=tags,
                attach_pdf=pdf_path is not None,
                pdf_path=pdf_path
            )
            
            self.saved_to_zotero.add(paper_id_str)
            
            # Move to next paper
            self.current_index += 1
            if self.current_index < len(self.results.results):
                return f"Saved to Zotero! Moving to paper {self.current_index + 1}...\n\n" + self._present_current_paper()
            else:
                return "Saved! That was the last paper. Review complete."
                
        except Exception as e:
            return f"Failed to save to Zotero: {e}"
    
    def _skip_current_paper(self) -> str:
        """Skip current paper and move to next."""
        self.current_index += 1
        
        if self.current_index < len(self.results.results):
            return f"Skipping. Moving to paper {self.current_index + 1}...\n\n" + self._present_current_paper()
        else:
            return "That was the last paper. Review complete."
    
    def _switch_to_paper(self, paper_index: int) -> str:
        """Switch to a different paper by index."""
        if paper_index < 0 or paper_index >= len(self.results.results):
            return f"Paper {paper_index + 1} doesn't exist. I have {len(self.results.results)} papers."
        
        self.current_index = paper_index
        return self._present_current_paper()
    
    def _exit_review(self) -> str:
        """Exit review mode."""
        saved_count = len(self.saved_to_zotero)
        return f"Review ended. Saved {saved_count} paper(s) to Zotero. Reviewed {self.current_index + 1} of {len(self.results.results)} papers."
    
    def _get_current_paper(self) -> Paper:
        """Get current paper being reviewed."""
        return self.results.results[self.current_index].paper
    
    def _add_to_history(self, message: str) -> None:
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })
        self.last_response = message
