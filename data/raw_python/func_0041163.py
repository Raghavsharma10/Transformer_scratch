def clean_markdown(text):
    """
    Parse markdown sintaxt to html.
    """
    result = text

    if isinstance(text, str):
        result = ''.join(
            BeautifulSoup(markdown(text), 'lxml').findAll(text=True))

    return result