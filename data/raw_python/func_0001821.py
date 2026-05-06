def html_to_ssml(text):
    """
    Replaces specific html tags with probable SSML counterparts.
    """
    ssml_text = reduce(lambda x, y: x.replace(y, html_to_ssml_maps[y]), html_to_ssml_maps, text)
    return ssml_text