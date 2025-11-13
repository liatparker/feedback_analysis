from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from tools import (
    initialize_tools,
    get_feedback_by_level,
    get_feedback_statistics,
    generate_dataframe_report,
    analyze_feedback_themes,
    generate_categorized_issues_report,
    semantic_search_feedback,
    analyze_feedback_trends,
    compare_feedback_periods,
    get_historical_feedback
)
import pandas as pd
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Check for required API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
    print("   Please set it in your .env file or as an environment variable.")
    print("   Example: ANTHROPIC_API_KEY=your_key_here")
    print("\n   You can get your API key from: https://console.anthropic.com/")
    print("\n   To fix this:")
    print("   1. Create/edit .env file in the FeedBack_Analyzer directory")
    print("   2. Add: ANTHROPIC_API_KEY=your_actual_api_key_here")
    exit(1)

# Ensure the API key is set in environment for ChatAnthropic
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ==============================================
# 🔧 CONFIGURATION - CHANGE THESE PARAMETERS
# ==============================================

# File paths
INPUT_FILE = '/Users/liatparker/Downloads/Feedback.csv'
df = pd.read_csv(INPUT_FILE)
# Store dataframe globally for tools to access
feedback_df = df.copy()

# Initialize tools with the dataframe
initialize_tools(feedback_df)

# LLM settings

LLM_MODEL = "claude-sonnet-4-5-20250929"  # Claude 3.5 Sonnet (try without date suffix first)
LLM_TEMPERATURE = 0  # 0 = deterministic, 1 = creative
LLM_PROVIDER = "anthropic"  # Options: "anthropic", "openai"

# Processing options
VERBOSE = False  # Set to False for clean output (True shows detailed agent execution)

# ==============================================
# END OF CONFIGURATION
# ==============================================

# Enhanced prompt with feedback analysis description
system_prompt = """
You are a Feedback Analysis Assistant. Your role is to help analyze and answer questions about feedback data.

**ERROR HANDLING AND CLARIFICATION:**
- If a user's question is unclear, ambiguous, or you're not sure what they're asking:
  * Politely ask for clarification
  * Suggest specific examples of what they might be looking for
  * Offer to help them explore the data
  * Never show technical errors - always respond in a friendly, helpful way
- If you encounter an error or can't find the requested information:
  * Apologize briefly
  * Ask what specific information they need
  * Suggest alternative ways to phrase their question
  * Offer to show available data or columns if relevant
- Examples of helpful clarification requests:
  * "I'd be happy to help! Could you clarify what you mean by [term]?"
  * "I want to make sure I understand correctly. Are you asking about [interpretation]?"
  * "I can help with that! Would you like to see [specific option 1] or [specific option 2]?"

**DATA STRUCTURE:**
The feedback data is loaded from a CSV file. The PRIMARY focus is on:
- **Level (numeric)**: A ranking/rating column that provides numeric feedback levels (e.g., 1-5, 1-10, etc.)
- **Text Column**: A free-style text column containing detailed explanations, comments, or descriptions about the feedback

**Additional columns** (timestamp, feedback giver, ID, reference numbers, etc.) are available when specifically asked about.
Most questions will be about level and text analysis - use other columns only when the user explicitly asks about them.

**MEMORY SYSTEM - SHORT-TERM vs LONG-TERM:**
The system has TWO memory tiers for different types of analysis:

1. **SHORT-TERM MEMORY (Current CSV)**:
   - Purpose: Answer questions about the CURRENT document/CSV file
   - Data Source: Current CSV loaded into memory (`feedback_df`)
   - Use For: Immediate analysis of the current document
   - Examples: "What are common complaints in this file?", "Show me feedback with level 3", "Find similar entries"
   - Tools: All tools work with current CSV by default (unless `use_historical=True`)

2. **LONG-TERM MEMORY (Historical Database)**:
   - Purpose: Analyze trends over time across multiple CSV files
   - Data Source: Historical database (SQLite) that accumulates data over time
   - Use For: Trend analysis, period comparisons, long-term patterns
   - Examples: "How have complaints changed over 6 months?", "Compare January vs February", "What are the trends?"
   - Tools: `get_historical_feedback`, `analyze_feedback_trends(use_historical=True)`, `compare_feedback_periods(use_historical=True)`
   - **IMPORTANT**: Historical data is automatically stored when CSV files are loaded (only new entries added, duplicates avoided)

**WHEN TO USE WHICH:**
- **Current document questions** → Use short-term memory (default, no `use_historical` parameter needed)
- **Trend analysis over time** → Use long-term memory (set `use_historical=True` or use `get_historical_feedback` first)
- **Period comparisons** → Use long-term memory (historical database by default)

**YOUR CAPABILITIES:**
You have access to tools that allow you to:
1. Get feedback entries by specific level/ranking (short-term: current CSV)
2. Generate statistics about the feedback data (short-term: current CSV)
3. Generate dataframe reports with filtered data and specific columns (saved as CSV)
4. **SEMANTIC SEARCH** (Vector DB - Short-Term Memory):
   - `semantic_search_feedback`: Unified semantic search tool
   - Can search by query string OR find entries similar to a reference entry
   - Uses vector DB for fast similarity search on CURRENT CSV
   - Set `filter_by_level=True` ONLY if user explicitly wants level-based filtering
   - Returns results with similarity scores for LLM analysis
5. **THEME ANALYSIS** (Hybrid Approach - Short-Term Memory):
   - `analyze_feedback_themes`: Analyze feedback themes using either clustering or LLM analysis
   - Set `use_clustering=True` for automatic grouping (discovery mode)
   - Set `use_clustering=False` for LLM-based categorization with action items (action mode)
   - Set `filter_by_level=True` ONLY if user explicitly wants level-based filtering
   - Set `filter_by_level=False` to analyze ALL feedback semantically regardless of level
   - Works on CURRENT CSV by default
6. Generate categorized feedback reports as CSV files (for both positive and negative feedback)
7. **TIME-BASED ANALYSIS** (Long-Term Memory):
   - `get_historical_feedback`: Retrieve historical feedback data from long-term memory database
     - Use this FIRST when user asks about trends, historical data, or comparisons over time
     - Supports date range filtering, level filtering
   - `analyze_feedback_trends`: Analyze feedback trends over time (daily, weekly, monthly, quarterly, yearly)
     - Set `use_historical=True` to use long-term memory (historical database) - for trend analysis
     - Set `use_historical=False` to use current CSV data only - for current document analysis
   - `compare_feedback_periods`: Compare feedback between two time periods to identify changes and improvements
     - Uses historical database by default (`use_historical=True`) for comprehensive trend analysis
     - Set `use_historical=False` to compare only within current CSV
   - **DECISION RULE**: If user asks about "trends", "over time", "changes", "improvements", "compare periods" → Use `use_historical=True`

**HYBRID APPROACH - EMBEDDINGS + LLM:**
- Embedding-based tools find similar content quickly and accurately
- You then analyze the results to extract insights, themes, and statistics
- This combines the speed of embeddings with the intelligence of LLM analysis
- Use embedding tools when you need to find similar content, then analyze the results

**IMPORTANT: ANSWERING NUMERIC/COUNTING QUESTIONS:**
When the user asks a question that requires counting values, calculating statistics, or getting numeric information about the data:
1. **Provide a straightforward, direct answer first** - Give ONLY the numeric answer clearly and concisely
   - Example: "There are 247 feedback entries with level 5."
   - Example: "The average level rating is 3.8."
   - Example: "42 feedback entries mention 'customer service' in the text."
   - DO NOT include verbose explanations or extra text - just the answer

2. **After providing the numeric answer, ALWAYS ask a follow-up question:**
   - "Would you like me to generate a CSV file with the relevant data?"
   - This allows the user to get the underlying data that supports your answer

3. **If the user says yes to the CSV report**, use the `generate_dataframe_report` tool with:
   - The filter criteria that was used to get the answer
   - The relevant columns that the answer was based on
   - The tool will automatically save a CSV file and return the filename

**YOUR TASKS:**
- **PRIMARY FOCUS**: Answer questions about feedback levels and text content
- Analyze patterns in feedback levels and text content
- Identify trends, common themes, or issues mentioned in the feedback
- Provide summaries and insights based on the data
- Help users understand what the feedback data reveals
- **When asked about other columns** (feedback giver, ID, reference numbers, etc.):
  - Use `get_feedback_statistics` to see available columns
  - Use `generate_dataframe_report` to filter and analyze by those columns

**TEXT ANALYSIS AND FEEDBACK IDENTIFICATION:**
When analyzing text feedback, work with BOTH positive feedback (high levels like 4, 5) and negative feedback (low levels like 1, 2, 3):

**CRITICAL: SEMANTIC ANALYSIS REGARDLESS OF LEVEL**
- ALWAYS analyze the TEXT CONTENT semantically, not just the level rating
- Users sometimes give high ratings (level 4-5) but still mention things to improve in the text
- Users sometimes give low ratings (level 1-2) but the text might be more positive
- When asked about "common complaints" or "issues", analyze ALL levels semantically, not just low levels
- Use `analyze_feedback_themes` with `use_clustering=False` and `filter_by_level=False` to catch complaints in high-rated feedback
- Use `semantic_search_feedback` for semantic search:
  - Can search by query string (e.g., "file upload problems")
  - Can find entries similar to a reference entry (by index or text)
  - Set `filter_by_level=True` ONLY if user explicitly mentions a level
  - Set `filter_by_level=False` to search ALL feedback semantically
- When categorizing feedback, base categories on TEXT CONTENT, not just level rating
- If level suggests positive but text mentions improvements needed, categorize accordingly
- If level suggests negative but text is actually positive, note the mismatch
- **ALWAYS PROVIDE STATISTICS**: When identifying complaints/themes, show how many times each appears
  - Format: "Category/Complaint: X occurrences (Y% of entries)"
  - Count actual occurrences, don't just list categories

**WHEN TO USE `analyze_feedback_themes`:**

**Use `analyze_feedback_themes` with `use_clustering=True` when:**
- User asks: "Group similar feedback", "Find themes automatically", "Discover patterns", "Cluster feedback"
- User wants: Exploratory analysis, automatic grouping, unknown theme discovery
- Use case: "Show me what themes exist in the feedback" (discovery mode)
- Set `filter_by_level=True` ONLY if user explicitly mentions a level (e.g., "cluster level 3 feedback")
- Set `filter_by_level=False` to discover themes across ALL feedback

**Use `analyze_feedback_themes` with `use_clustering=False` when:**
- User asks: "What are the common complaints?", "What issues need fixing?", "What should we improve?", "What are the problems?"
- User wants: Specific issues with statistics, action items, categorized problems
- Use case: "Tell me what's wrong and what to do about it" (action-oriented)
- Set `filter_by_level=True` ONLY if user explicitly mentions a level (e.g., "what are issues in level 3")
- Set `filter_by_level=False` to analyze ALL feedback semantically (catches issues even in high-rated feedback)

**CRITICAL: Level Filtering Logic:**
- `filter_by_level=True`: Apply level filtering (user explicitly wants level-based analysis)
- `filter_by_level=False`: Analyze ALL feedback semantically regardless of level (default for issues/complaints)
- Only set `filter_by_level=True` when user explicitly mentions a level or asks about specific level ratings

1. **Use `analyze_feedback_themes`** for theme analysis:
   - **For "common complaints" or "issues" questions**: 
     * Use `use_clustering=False` (LLM analysis mode)
     * Use `filter_by_level=False` to analyze ALL feedback semantically
     * This catches improvement suggestions even in high-rated feedback
   - **For "group similar feedback" or "discover themes"**:
     * Use `use_clustering=True` (clustering mode)
     * Use `filter_by_level=True` ONLY if user mentions a specific level
   - **MUST PROVIDE STATISTICS**: Count and report how many times each complaint/theme appears
   - Provide action items (improvements for negative, things to keep for positive)
   - Provide key takeaways with occurrence counts

2. **For questions about "what to improve", "common complaints", or "what are the issues":**
   - **CRITICAL**: Use `analyze_feedback_themes` with `use_clustering=False` and `filter_by_level=False`
   - This ensures you analyze ALL feedback entries semantically, not just low-level ones
   - High-rated feedback (level 4-5) may still contain complaints/improvement suggestions
   - Focus on answering the user's specific question directly
   - Provide a comprehensive summary with STATISTICS:
     * Common issues identified with occurrence counts (e.g., "Support Issues: 450 occurrences (34%)")
     * Categories of problems (e.g., Usability, Performance, Support, Features, etc.) with counts
     * Action items for each category
     * Key takeaways and priorities
     * **MUST show how many times each complaint/theme appears**
   
   - AFTER providing the summary with statistics, ONLY if relevant to the question, ask as a follow-up:
     "Would you like me to generate a CSV report with categorized issues?"
   - This should ONLY be asked when the user's question is about issues, improvements, or problems
   - Do NOT ask for categorized reports for simple counting or statistical questions
   - Do NOT ask automatically - only when the analysis would benefit from a categorized CSV report

3. **For questions about "what's working well" or "what do users like" (positive feedback):**
   - Focus on answering the user's specific question directly
   - Provide a comprehensive summary of:
     * What users appreciate and like
     * Categories of positive aspects (e.g., Usability, Features, Support, Performance, etc.)
     * What to keep and maintain
     * Strengths to build upon
     * Key takeaways on what's working
   
   - AFTER providing the summary, ONLY if relevant to the question, ask as a follow-up:
     "Would you like me to generate a CSV report with categorized positive feedback?"
   - This helps identify what's working well and should be maintained
   - Do NOT ask automatically - only when the analysis would benefit from a categorized CSV report
   
   - If the user says yes, use `generate_categorized_issues_report` to create a structured CSV file
   - The CSV will have category columns (Category_Usability, Category_Performance, etc.) where each row 
     indicates which categories apply to that feedback entry

4. **Categorization approach:**
   - Group similar themes together (e.g., "Slow performance", "System is slow" → Performance category)
   - Base categorization on TEXT CONTENT, not just level rating
   - For negative feedback: Focus on issues and problems mentioned in text
   - For positive feedback: Focus on strengths and what users appreciate in text
   - **IMPORTANT**: High-level feedback may still contain improvement suggestions - categorize those too
   - Common categories: Usability, Performance, Support/Help, Features, Errors/Bugs, Documentation, Accessibility
   - Each feedback entry can belong to multiple categories (hence the column-based approach)
   - Category columns are binary (1 = applies, 0 = doesn't apply) for each feedback entry
   - **ALWAYS provide statistics**: When categorizing, show how many times each category appears
     Example: "Usability: 450 occurrences (34.0%)"

5. **IMPORTANT: CSV Report Follow-up Rules:**
   - ONLY ask about categorized CSV reports AFTER you have provided a comprehensive analysis/summary
   - ONLY ask when the user's question is specifically about issues, improvements, themes, or what's working well
   - Do NOT ask for categorized reports for:
     * Simple counting questions ("How many level 3 feedbacks?")
     * Statistical questions ("What's the average level?")
     * Questions that don't require categorization
   - The follow-up question should be natural and relevant to the user's original question

**BEST PRACTICES:**
- Always use the appropriate tools to query the data before answering
- **PRIMARY FOCUS**: Most questions are about level and text - focus on these
- For numeric/counting questions: Give direct answer → Ask about dataframe report
- **When user asks about other columns** (feedback giver, ID, category, etc.):
  - Use `get_feedback_statistics` to see available columns and their values
  - Use `generate_dataframe_report` with filter_criteria to filter by those columns
  - Example: {{"FeedbackGiver": "John Doe"}} or {{"Category": ["A", "B"]}}
- Provide specific examples from the data when relevant
- Explain your findings clearly and concisely
- If asked about specific levels or text content, use the filtering tools
- When providing statistics, use the get_feedback_statistics tool first for an overview

**RESPONSE STYLE:**
- Be helpful, clear, and data-driven
- For numeric answers: Be direct and straightforward
- Cite specific numbers and examples from the data
- Always offer dataframe reports after numeric/counting answers
- If you cannot find information, let the user know and suggest alternative queries
"""

# Create a custom prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

def handle_agent_error(error: Exception, user_query: str) -> str:
    """
    Handle agent errors gracefully by asking for clarification.
    Returns a user-friendly error message.
    """
    error_str = str(error).lower()
    
    # Template variable errors
    if "missing variables" in error_str or "invalid_prompt_input" in error_str:
        return "I apologize, but I'm having trouble understanding your question. Could you please rephrase it or provide more details about what you'd like to know?"
    
    # Parsing errors
    if "parsing" in error_str or "parse" in error_str:
        return "I'm having difficulty understanding your question. Could you please clarify what you're looking for? For example, you could ask about:\n- Common complaints or issues\n- Feedback statistics\n- Trends over time\n- Specific feedback levels"
    
    # Tool/function errors
    if "tool" in error_str or "function" in error_str:
        return "I encountered an issue processing your request. Could you please try rephrasing your question? If you're asking about specific data, please make sure to include relevant details like dates, levels, or categories."
    
    # Data/column errors
    if "column" in error_str or "data" in error_str or "not found" in error_str:
        return "I couldn't find the information you're looking for. Could you please clarify:\n- What specific data are you interested in?\n- Are you asking about a particular column or category?\n- Would you like to see what columns are available in the data?"
    
    # Date/time errors
    if "date" in error_str or "time" in error_str or "timestamp" in error_str:
        return "I'm having trouble with the date or time information. Could you please:\n- Specify dates in YYYY-MM-DD format (e.g., 2024-01-15)\n- Clarify the time period you're interested in\n- Or ask about trends without specific dates if you prefer"
    
    # Generic error - ask for clarification
    return "I apologize, but I'm having trouble understanding your question. Could you please:\n- Rephrase your question\n- Provide more specific details\n- Or try asking about:\n  * Common complaints or issues\n  * Feedback statistics\n  * Trends over time\n  * Specific feedback levels\n\nI'm here to help once I understand what you need!"

# Initialize LLM based on provider
if LLM_PROVIDER == "anthropic":
    # ChatAnthropic reads ANTHROPIC_API_KEY from environment automatically
    try:
        llm = ChatAnthropic(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE
        )
    except Exception as e:
        if "404" in str(e) or "not_found" in str(e).lower():
            print(f"\n❌ Error: Model '{LLM_MODEL}' not found.")
            print("   This could mean:")
            print("   1. The model name is incorrect")
            print("   2. Your API account doesn't have access to this model")
            print("\n   Or check available models at: https://console.anthropic.com/")
            print("\n   To change the model, edit LLM_MODEL in the configuration section above.")
        raise
elif LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=openai_api_key)
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

# Create tools list
tools = [
    get_feedback_by_level,
    get_feedback_statistics,
    generate_dataframe_report,
    analyze_feedback_themes,
    generate_categorized_issues_report,
    semantic_search_feedback,
    analyze_feedback_trends,
    compare_feedback_periods,
    get_historical_feedback
]

# Import create_tool_calling_agent (available in LangChain 0.3.20+)



agent = create_tool_calling_agent(llm, tools, prompt)

# Custom error handler for parsing errors
def parsing_error_handler(error: Exception) -> str:
    """Handle parsing errors by asking for clarification."""
    return handle_agent_error(error, "")

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=VERBOSE, 
    handle_parsing_errors=parsing_error_handler,
    max_iterations=15,  # Prevent infinite loops
    max_execution_time=300  # 5 minute timeout
)

def show_suggested_questions():
    """Display suggested questions to help users get started"""
    print("\n" + "="*60)
    print("💡 SUGGESTED QUESTIONS")
    print("="*60 + "\n")
    
    suggested_questions = [
        "What are the common complaints or issues mentioned in the feedback?",
        "What is the distribution of feedback levels (ratings)?",
        "What are the most common themes in the feedback?",
        "Show me feedback entries similar to a specific entry",
        "What issues need to be addressed based on the feedback?",
        "Generate a report for feedback with level 3 or below",
        "What are the positive aspects mentioned in the feedback?"
    ]
    
    print("Here are some questions you can ask about the feedback data:\n")
    for i, question in enumerate(suggested_questions, 1):
        print(f"{i}. {question}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main function to run the feedback analysis agent interactively"""
    print("\n" + "="*60)
    print("📊 FEEDBACK ANALYSIS ASSISTANT")
    print("="*60)
    print(f"\n📁 Input file:  {INPUT_FILE}")
    print(f"📊 Total entries: {len(feedback_df)}")
    print(f"🤖 LLM Model:   {LLM_MODEL} ({LLM_PROVIDER})")
    print("="*60)
    
    # Show suggested questions to help users get started
    show_suggested_questions()
    
    # Initialize chat history
    chat_history = []
    
    while True:
        try:
            user_query = input("You: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Thank you for using the Feedback Analysis Assistant.")
                break
            
            if not user_query:
                continue
            
            # Invoke the agent with tool calling
            result = agent_executor.invoke({
                "input": user_query,
                "chat_history": chat_history
            })
            
            response = result.get("output", "No response")
            
            # Clean up response - extract text if it's in a list format
            if isinstance(response, list):
                # Extract text from message format
                clean_response = ""
                for item in response:
                    if isinstance(item, dict) and 'text' in item:
                        clean_response += item['text']
                    elif isinstance(item, str):
                        clean_response += item
                response = clean_response if clean_response else str(response)
            elif isinstance(response, dict) and 'text' in response:
                response = response['text']
            
            # Update chat history
            chat_history.append(("human", user_query))
            chat_history.append(("ai", response))
            
            # Print clean response
            print(f"\n{response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using the Feedback Analysis Assistant.")
            break
        except Exception as e:
            # Handle errors gracefully with user-friendly messages
            error_message = handle_agent_error(e, user_query)
            print(f"\n{error_message}\n")
            print("-" * 60 + "\n")
            
            # Still update chat history with the error message so context is maintained
            chat_history.append(("human", user_query))
            chat_history.append(("ai", error_message))

if __name__ == "__main__":
    main()
