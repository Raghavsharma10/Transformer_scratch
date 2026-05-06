def loads(s, class_=None, **kwargs):
    """
    Convert content in a JSON-encoded string to a Physical Information Object or a list of such objects.

    :param s: String to deserialize.
    :param class_: Subclass of :class:`.Pio` to produce, if not unambiguous
    :param kwargs: Any options available to json.loads().
    :return: Single object derived from :class:`.Pio` or a list of such object.
    """
    return loado(json.loads(s, **kwargs), class_=class_)