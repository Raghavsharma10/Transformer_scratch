def _parse_postmeta(element):
    import phpserialize

    """
    Retrive post metadata as a dictionary
    """

    metadata = {}
    fields = element.findall("./{%s}postmeta" % WP_NAMESPACE)

    for field in fields:
        key = field.find("./{%s}meta_key" % WP_NAMESPACE).text
        value = field.find("./{%s}meta_value" % WP_NAMESPACE).text

        if key == "_wp_attachment_metadata":
            stream = StringIO(value.encode())
            try:
                data = phpserialize.load(stream)
                metadata["attachment_metadata"] = data
            except ValueError as e:
                pass
            except Exception as e:
                raise(e)

        if key == "_wp_attached_file":
            metadata["attached_file"] = value

    return metadata