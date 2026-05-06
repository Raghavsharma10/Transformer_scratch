def _parse(root):
    """Recursively convert an Element into python data types"""
    if root.tag == "nil-classes":
        return []
    elif root.get("type") == "array":
        return [_parse(child) for child in root]

    d = {}
    for child in root:
        type = child.get("type") or "string"

        if child.get("nil"):
            value = None
        elif type == "boolean":
            value = True if child.text.lower() == "true" else False
        elif type == "dateTime":
            value = iso8601.parse_date(child.text)
        elif type == "decimal":
            value = decimal.Decimal(child.text)
        elif type == "integer":
            value = int(child.text)
        else:
            value = child.text

        d[child.tag] = value
    return d