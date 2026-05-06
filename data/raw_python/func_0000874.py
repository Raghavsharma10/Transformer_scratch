def build_doctype(qualifiedName, publicId=None, systemId=None, internalSubset=None):
    """
    Instantiate an ElifeDocumentType, a subclass of minidom.DocumentType, with
    some properties so it is more testable
    """
    doctype = ElifeDocumentType(qualifiedName)
    doctype._identified_mixin_init(publicId, systemId)
    if internalSubset:
        doctype.internalSubset = internalSubset
    return doctype