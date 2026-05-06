def parseMarkDownBlock(text):
    """
    Parses a block of text, returning a list of docutils nodes

    >>> parseMarkdownBlock("Some\n====\n\nblock of text\n\nHeader\n======\n\nblah\n")
    []
    """
    block = Parser().parse(text)
    # CommonMark can't nest sections, so do it manually
    nestSections(block)

    return MarkDown(block)