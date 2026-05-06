def htmlize_paragraphs(text):
    """
    Convert paragraphs delimited by blank lines into HTML text enclosed
    in <p> tags.
    """
    paragraphs = re.split('(\r?\n)\s*(\r?\n)', text)
    return '\n'.join('<p>%s</p>' % paragraph for paragraph in paragraphs)