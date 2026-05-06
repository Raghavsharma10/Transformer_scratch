def validate_doc(schema_names,  # type: Names
                 doc,           # type: Union[Dict[Text, Any], List[Dict[Text, Any]], Text, None]
                 loader,        # type: Loader
                 strict,        # type: bool
                 strict_foreign_properties=False  # type: bool
                 ):
    # type: (...) -> None
    """Validate a document using the provided schema."""
    has_root = False
    for root in schema_names.names.values():
        if ((hasattr(root, 'get_prop') and root.get_prop(u"documentRoot")) or (
                u"documentRoot" in root.props)):
            has_root = True
            break

    if not has_root:
        raise validate.ValidationException(
            "No document roots defined in the schema")

    if isinstance(doc, MutableSequence):
        vdoc = doc
    elif isinstance(doc, CommentedMap):
        vdoc = CommentedSeq([doc])
        vdoc.lc.add_kv_line_col(0, [doc.lc.line, doc.lc.col])
        vdoc.lc.filename = doc.lc.filename
    else:
        raise validate.ValidationException("Document must be dict or list")

    roots = []
    for root in schema_names.names.values():
        if ((hasattr(root, "get_prop") and root.get_prop(u"documentRoot")) or (
                root.props.get(u"documentRoot"))):
            roots.append(root)

    anyerrors = []
    for pos, item in enumerate(vdoc):
        sourceline = SourceLine(vdoc, pos, Text)
        success = False
        for root in roots:
            success = validate.validate_ex(
                root, item, loader.identifiers, strict,
                foreign_properties=loader.foreign_properties,
                raise_ex=False, skip_foreign_properties=loader.skip_schemas,
                strict_foreign_properties=strict_foreign_properties)
            if success:
                break

        if not success:
            errors = []  # type: List[Text]
            for root in roots:
                if hasattr(root, "get_prop"):
                    name = root.get_prop(u"name")
                elif hasattr(root, "name"):
                    name = root.name

                try:
                    validate.validate_ex(
                        root, item, loader.identifiers, strict,
                        foreign_properties=loader.foreign_properties,
                        raise_ex=True, skip_foreign_properties=loader.skip_schemas,
                        strict_foreign_properties=strict_foreign_properties)
                except validate.ClassValidationException as exc:
                    errors = [sourceline.makeError(u"tried `%s` but\n%s" % (
                        name, validate.indent(str(exc), nolead=False)))]
                    break
                except validate.ValidationException as exc:
                    errors.append(sourceline.makeError(u"tried `%s` but\n%s" % (
                        name, validate.indent(str(exc), nolead=False))))

            objerr = sourceline.makeError(u"Invalid")
            for ident in loader.identifiers:
                if ident in item:
                    objerr = sourceline.makeError(
                        u"Object `%s` is not valid because"
                        % (relname(item[ident])))
                    break
            anyerrors.append(u"%s\n%s" %
                             (objerr, validate.indent(bullets(errors, "- "))))
    if anyerrors:
        raise validate.ValidationException(
            strip_dup_lineno(bullets(anyerrors, "* ")))