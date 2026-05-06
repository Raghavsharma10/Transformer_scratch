def list_spectrum_descriptions(*args):
    """List all available spectrum descriptions that can be used for peak list simulation.

    :param str args: Spectrum name(s), e.g. list_spectrum_descriptions("HNCO", "HNcoCACB"), leave empty to list everything.
    :return: None
    :rtype: :py:obj:`None`
    """
    if args:
        for spectrum_name in args:
            pprint.pprint({spectrum_name: SPECTRUM_DESCRIPTIONS.get(spectrum_name, None)}, width=120)
    else:
        pprint.pprint(SPECTRUM_DESCRIPTIONS, width=120)