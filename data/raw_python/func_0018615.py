def full_split(text, regex):
    """
    Split the text by the regex, keeping all parts.
    The parts should re-join back into the original text.

    >>> list(full_split('word', re.compile('&.*?')))
    ['word']
    """
    while text:
        m = regex.search(text)
        if not m:
            yield text
            break
        left = text[:m.start()]
        middle = text[m.start():m.end()]
        right = text[m.end():]
        if left:
            yield left
        if middle:
            yield middle
        text = right