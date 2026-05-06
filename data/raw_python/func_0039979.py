def pdf_to_text(pdf_filepath='', **kwargs):
    """
    Parse pdf to a list of strings using the pdfminer lib.

    Args:
        no_laparams=False,
        all_texts=None,
        detect_vertical=None, word_margin=None, char_margin=None,
        line_margin=None, boxes_flow=None, codec='utf-8',
        strip_control=False, maxpages=0, page_numbers=None, password="",
        scale=1.0, rotation=0, layoutmode='normal', debug=False,
        disable_caching=False,
    """

    result = []
    try:
        if not os.path.exists(pdf_filepath):
            raise ValueError("No valid pdf filepath introduced..")

        # TODO: REVIEW THIS PARAMS
        # update params if not defined
        kwargs['outfp'] = kwargs.get('outfp', StringIO())
        kwargs['laparams'] = kwargs.get('laparams', pdfminer.layout.LAParams())
        kwargs['imagewriter'] = kwargs.get('imagewriter', None)
        kwargs['output_type'] = kwargs.get('output_type', "text")
        kwargs['codec'] = kwargs.get('codec', 'utf-8')
        kwargs['disable_caching'] = kwargs.get('disable_caching', False)

        with open(pdf_filepath, "rb") as f_pdf:
            pdfminer.high_level.extract_text_to_fp(f_pdf, **kwargs)

        result = kwargs.get('outfp').getvalue()

    except Exception:
        logger.error('fail pdf to text parsing')

    return result