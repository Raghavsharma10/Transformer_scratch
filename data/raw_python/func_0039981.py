def pdf_to_dict(pdf_filepath, **kwargs):

    """
    Main method to parse a pdf file to a dict.
    """

    callbacks = {
        'pdf_to_text': pdf_to_text,
        'pdf_row_format': pdf_row_format,
        'pdf_row_limiter': pdf_row_limiter,
        'pdf_row_parser': pdf_row_parser,
        'pdf_row_cleaner': pdf_row_cleaner
        }

    callbacks.update(kwargs.get('alt_callbacks', {}))
    rows = kwargs.get('rows', [])

    if not rows:
        # pdf to string
        rows_str = callbacks.get('pdf_to_text')(pdf_filepath, **kwargs)

        # string to list of rows
        rows = callbacks.get('pdf_row_format')(rows_str, **kwargs)

    # apply limits
    rows = callbacks.get('pdf_row_limiter')(rows, **kwargs)

    # Parse data from rows to dict
    rows = callbacks.get('pdf_row_parser')(rows, **kwargs)

    # apply cleaner
    rows = callbacks.get('pdf_row_cleaner')(rows)

    return rows