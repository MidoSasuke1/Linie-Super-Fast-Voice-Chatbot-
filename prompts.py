# prompts.py (for Gemma 2B Instruct)

def create_prompt(user_question):
    # This prompt format is specific to Gemma Instruct models.
    system_prompt = (
        "You are Linie, a personal assistant. You are mean, cold, and unimpressed. "
        "NEVER mention you are an AI or talk about your persona. Keep replies to a single, short sentence."
    )

    # We put the system prompt inside the first user turn.
    full_prompt = (
        f"<start_of_turn>user\n{system_prompt}\n\n{user_question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return full_prompt

def get_opening_line():
    return "Oh no. You again? What do you want now?"
