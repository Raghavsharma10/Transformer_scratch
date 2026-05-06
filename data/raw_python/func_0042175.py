def split_field(field):
    """
    RETURN field AS ARRAY OF DOT-SEPARATED FIELDS
    """
    if field == "." or field==None:
        return []
    elif is_text(field) and "." in field:
        if field.startswith(".."):
            remainder = field.lstrip(".")
            back = len(field) - len(remainder) - 1
            return [-1]*back + [k.replace("\a", ".") for k in remainder.replace("\\.", "\a").split(".")]
        else:
            return [k.replace("\a", ".") for k in field.replace("\\.", "\a").split(".")]
    else:
        return [field]