def set_document_type(loader_cls: Type, type_: Type) -> None:
    """Set the type corresponding to the whole document.

    Args:
        loader_cls: The loader class to set the document type for.
        type_: The type to loader should process the document into.
    """
    loader_cls.document_type = type_

    if not hasattr(loader_cls, '_registered_classes'):
        loader_cls._registered_classes = dict()