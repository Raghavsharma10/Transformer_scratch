def clean_text(text):
    """
    Retrieve clean text without markdown sintax or other things.
    """
    if text:
        text = html2text.html2text(clean_markdown(text))
        return re.sub(r'\s+', ' ', text).strip()