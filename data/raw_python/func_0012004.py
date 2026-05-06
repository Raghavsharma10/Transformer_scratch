def _unicode_to_ascii(obj):  # pragma: no cover
    """Convert to ASCII."""
    # pylint: disable=E0602,R1717
    if isinstance(obj, dict):
        return dict(
            [
                (_unicode_to_ascii(key), _unicode_to_ascii(value))
                for key, value in obj.items()
            ]
        )
    if isinstance(obj, list):
        return [_unicode_to_ascii(element) for element in obj]
    if isinstance(obj, unicode):
        return obj.encode("utf-8")
    return obj