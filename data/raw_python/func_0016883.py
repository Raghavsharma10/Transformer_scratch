def dumped(text, level, indent=2):
    """Put curly brackets round an indented text"""
    return indented("{\n%s\n}" % indented(text, level + 1, indent) or "None", level, indent) + "\n"