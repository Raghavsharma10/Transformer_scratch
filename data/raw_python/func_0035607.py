def shell_escape(text, _safe=re.compile(r"^[-._,+a-zA-Z0-9]+$")):
    """Escape given string according to shell rules."""
    if not text or _safe.match(text):
        return text

    squote = type(text)("'")
    return squote + text.replace(squote, type(text)(r"'\''")) + squote