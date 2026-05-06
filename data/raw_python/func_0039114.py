def _prompt_username(prompt="Username: ", prefill=None):
    """Prompt the user for username."""
    if prefill:
        readline.set_startup_hook(lambda: readline.insert_text(prefill))

    try:
        return input(prompt).strip()
    except EOFError:
        print()
    finally:
        readline.set_startup_hook()