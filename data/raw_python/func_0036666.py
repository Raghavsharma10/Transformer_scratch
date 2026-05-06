def extract_ids(text, extractors):
    """
    Uses `extractors` to extract citation identifiers from a text.

    :Parameters:
        text : str
            The text to process
        extractors : `list`(`extractor`)
            A list of extractors to apply to the text

    :Returns:
        `iterable` -- a generator of extracted identifiers
    """
    for extractor in extractors:
        for id in extractor.extract(text):
            yield id