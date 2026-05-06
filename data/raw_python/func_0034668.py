def find(whatever=None, language=None, iso639_1=None,
         iso639_2=None, native=None):
    """Find data row with the language.

    :param whatever: key to search in any of the following fields
    :param language: key to search in English language name
    :param iso639_1: key to search in ISO 639-1 code (2 digits)
    :param iso639_2: key to search in ISO 639-2 code (3 digits,
                     bibliographic & terminological)
    :param native: key to search in native language name
    :return: a dict with keys (u'name', u'iso639_1', u'iso639_2_b',
                     u'iso639_2_t', u'native')

    All arguments can be both string or unicode (Python 2).
    If there are multiple names defined, any of these can be looked for.
    """
    if whatever:
        keys = [u'name', u'iso639_1', u'iso639_2_b', u'iso639_2_t', u'native']
        val = whatever
    elif language:
        keys = [u'name']
        val = language
    elif iso639_1:
        keys = [u'iso639_1']
        val = iso639_1
    elif iso639_2:
        keys = [u'iso639_2_b', u'iso639_2_t']
        val = iso639_2
    elif native:
        keys = [u'native']
        val = native
    else:
        raise ValueError('Invalid search criteria.')
    val = unicode(val).lower()
    return next((item for item in data if any(
        val in item[key].lower().split("; ") for key in keys)), None)