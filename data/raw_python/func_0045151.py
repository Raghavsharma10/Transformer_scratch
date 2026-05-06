def parse_owl_xml(url):
    """Downloads and parses an OWL resource in OWL/XML format using the :class:`OWLParser`.

    :param str url: The URL to the OWL resource
    :return: A directional graph representing the OWL document's hierarchy
    :rtype: networkx.DiGraph
    """
    res = download(url)
    owl = OWLParser(content=res.content)
    return owl