"""
Rule-Based AI Chatbot
A deterministic, dictionary-driven conversational agent using O(1) intent lookups.
"""

import random
import datetime

# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------
# Each intent maps to one or more possible responses for natural variation.
KNOWLEDGE_BASE = {
    "hello": ["Hi there! How can I help you today?", "Hello! What can I do for you?"],
    "hi": ["Hey! How's it going?", "Hi there!"],
    "how are you": ["I'm just a program, but I'm running smoothly! How about you?"],
    "what is your name": ["I'm a rule-based chatbot built for DecodeLabs Project 1."],
    "who made you": ["I was built as part of the DecodeLabs AI Engineering track."],
    "what can you do": ["I can respond to greetings, answer simple questions, and chat with you using predefined rules."],
    "time": [],   # handled dynamically below
    "date": [],   # handled dynamically below
    "help": ["You can ask me things like 'hello', 'how are you', 'what is your name', or type 'exit' to quit."],
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["No problem!", "Glad to help!"],
    "bye": ["Goodbye! Have a great day."],
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

FALLBACK_RESPONSES = [
    "I do not understand that. Could you rephrase?",
    "I'm not sure how to respond to that yet.",
    "Sorry, I don't have a rule for that input.",
]


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
def sanitize_input(raw_input: str) -> str:
    """Normalize user input: strip whitespace, lowercase, remove trailing punctuation."""
    cleaned = raw_input.strip().lower()
    cleaned = cleaned.rstrip("!?.,")
    return cleaned


def get_dynamic_response(intent: str) -> str:
    """Handle intents that require runtime-computed values."""
    if intent == "time":
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
    if intent == "date":
        return f"Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}."
    return None


def generate_response(user_input: str) -> str:
    """Match sanitized input against the knowledge base and return a response."""
    intent = sanitize_input(user_input)

    dynamic_reply = get_dynamic_response(intent)
    if dynamic_reply:
        return dynamic_reply

    if intent in KNOWLEDGE_BASE and KNOWLEDGE_BASE[intent]:
        return random.choice(KNOWLEDGE_BASE[intent])

    return random.choice(FALLBACK_RESPONSES)


def is_exit_command(user_input: str) -> bool:
    return sanitize_input(user_input) in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def run_chatbot():
    print("Chatbot: Hello! Type 'exit' or 'bye' anytime to end the conversation.")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Session terminated.")
            break

        if not user_input.strip():
            print("Chatbot: Please type something.")
            continue

        if is_exit_command(user_input):
            print("Chatbot: Goodbye! Have a great day.")
            break

        response = generate_response(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    run_chatbot()
