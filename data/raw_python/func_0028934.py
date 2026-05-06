def load(fp, class_=None, **kwargs):
    """
    Convert content in a JSON-encoded text file to a Physical Information Object or a list of such objects.

    :param fp: File-like object supporting .read() method to deserialize from.
    :param class_: Subclass of :class:`.Pio` to produce, if not unambiguous
    :param kwargs: Any options available to json.load().
    :return: Single object derived from :class:`.Pio` or a list of such object.
    """
    return loado(json.load(fp, **kwargs), class_=class_)