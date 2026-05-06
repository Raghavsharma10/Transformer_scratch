def load_schema(schema_ref,  # type: Union[CommentedMap, CommentedSeq, Text]
                cache=None   # type: Dict
                ):
    # type: (...) -> Tuple[Loader, Union[Names, SchemaParseException], Dict[Text, Any], Loader]
    """
    Load a schema that can be used to validate documents using load_and_validate.

    return: document_loader, avsc_names, schema_metadata, metaschema_loader
    """

    metaschema_names, _metaschema_doc, metaschema_loader = get_metaschema()
    if cache is not None:
        metaschema_loader.cache.update(cache)
    schema_doc, schema_metadata = metaschema_loader.resolve_ref(schema_ref, "")

    if not isinstance(schema_doc, MutableSequence):
        raise ValueError("Schema reference must resolve to a list.")

    validate_doc(metaschema_names, schema_doc, metaschema_loader, True)
    metactx = schema_metadata.get("@context", {})
    metactx.update(collect_namespaces(schema_metadata))
    schema_ctx = jsonld_context.salad_to_jsonld_context(schema_doc, metactx)[0]

    # Create the loader that will be used to load the target document.
    document_loader = Loader(schema_ctx, cache=cache)

    # Make the Avro validation that will be used to validate the target
    # document
    avsc_names = make_avro_schema(schema_doc, document_loader)

    return document_loader, avsc_names, schema_metadata, metaschema_loader