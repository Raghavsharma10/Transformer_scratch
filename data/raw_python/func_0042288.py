def expand(doc, doc_url="param://", params=None):
    """
    ASSUMING YOU ALREADY PULED THE doc FROM doc_url, YOU CAN STILL USE THE
    EXPANDING FEATURE

    USE mo_json_config.expand({}) TO ASSUME CURRENT WORKING DIRECTORY

    :param doc: THE DATA STRUCTURE FROM JSON SOURCE
    :param doc_url: THE URL THIS doc CAME FROM (DEFAULT USES params AS A DOCUMENT SOURCE)
    :param params: EXTRA PARAMETERS NOT FOUND IN THE doc_url PARAMETERS (WILL SUPERSEDE PARAMETERS FROM doc_url)
    :return: EXPANDED JSON-SERIALIZABLE STRUCTURE
    """
    if doc_url.find("://") == -1:
        Log.error("{{url}} must have a prototcol (eg http://) declared", url=doc_url)

    url = URL(doc_url)
    url.query = set_default(url.query, params)
    phase1 = _replace_ref(doc, url)  # BLANK URL ONLY WORKS IF url IS ABSOLUTE
    phase2 = _replace_locals(phase1, [phase1])
    return wrap(phase2)