def parse(filename, return_doctype_dict=False):
    """
    to extract the doctype details from the file when parsed and return the data
    for later use, set return_doctype_dict to True
    """
    doctype_dict = {}
    # check for python version, doctype in ElementTree is deprecated 3.2 and above
    if sys.version_info < (3,2):
        parser = CustomXMLParser(html=0, target=None, encoding='utf-8')
    else:
        # Assume greater than Python 3.2, get the doctype from the TreeBuilder
        tree_builder = CustomTreeBuilder()
        parser = ElementTree.XMLParser(html=0, target=tree_builder, encoding='utf-8')

    tree = ElementTree.parse(filename, parser)
    root = tree.getroot()

    if sys.version_info < (3,2):
        doctype_dict = parser.doctype_dict
    else:
        doctype_dict = tree_builder.doctype_dict

    if return_doctype_dict is True:
        return root, doctype_dict
    else:
        return root