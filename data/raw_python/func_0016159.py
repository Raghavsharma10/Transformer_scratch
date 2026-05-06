def load_and_validate(document_loader,                 # type: Loader
                      avsc_names,                      # type: Names
                      document,                        # type: Union[CommentedMap, Text]
                      strict,                          # type: bool
                      strict_foreign_properties=False  # type: bool
                      ):
    # type: (...) -> Tuple[Any, Dict[Text, Any]]
    """Load a document and validate it with the provided schema.

    return data, metadata
    """
    try:
        if isinstance(document, CommentedMap):
            data, metadata = document_loader.resolve_all(
                document, document["id"], checklinks=True,
                strict_foreign_properties=strict_foreign_properties)
        else:
            data, metadata = document_loader.resolve_ref(
                document, checklinks=True,
                strict_foreign_properties=strict_foreign_properties)

        validate_doc(avsc_names, data, document_loader, strict,
                     strict_foreign_properties=strict_foreign_properties)

        return data, metadata
    except validate.ValidationException as exc:
        raise validate.ValidationException(strip_dup_lineno(str(exc)))