def ascii_text(text):
    """Transliterate the given text and make sure it ends up as ASCII."""
    text = latinize_text(text, ascii=True)
    if isinstance(text, six.text_type):
        text = text.encode('ascii', 'ignore').decode('ascii')
    return text