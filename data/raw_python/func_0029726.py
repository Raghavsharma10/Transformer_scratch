def markdown(text):
    """Processes GFM then converts it to HTML."""
    text = gfm(text)
    text = markdown_lib.markdown(text)
    return text