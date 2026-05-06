def tail_field(field):
    """
    RETURN THE FIRST STEP IN PATH, ALONG WITH THE REMAINING TAIL
    """
    if field == "." or field==None:
        return ".", "."
    elif "." in field:
        if "\\." in field:
            return tuple(k.replace("\a", ".") for k in field.replace("\\.", "\a").split(".", 1))
        else:
            return field.split(".", 1)
    else:
        return field, "."