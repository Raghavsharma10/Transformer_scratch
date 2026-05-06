def text2vocab(text, output_file, text2wfreq_kwargs={}, wfreq2vocab_kwargs={}):
    """
        Convienience function that uses text2wfreq and wfreq2vocab to create a vocabulary file from text.
    """
    with tempfile.NamedTemporaryFile(suffix='.wfreq', delete=False) as f:
        wfreq_file = f.name

    try:
        text2wfreq(text, wfreq_file, **text2wfreq_kwargs)
        wfreq2vocab(wfreq_file, output_file, **wfreq2vocab_kwargs)
    except ConversionError:
        raise
    finally:
        os.remove(wfreq_file)