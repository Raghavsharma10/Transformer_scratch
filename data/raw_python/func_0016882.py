def indented(text, level, indent=2):
    """Take a multiline text and indent it as a block"""
    return "\n".join("%s%s" % (level * indent * " ", s) for s in text.splitlines())