def box(script):
    """
    :param es_script:
    :return: TEXT EXPRESSION WITH NON OBJECTS BOXED
    """
    if script.type is BOOLEAN:
        return "Boolean.valueOf(" + text_type(script.expr) + ")"
    elif script.type is INTEGER:
        return "Integer.valueOf(" + text_type(script.expr) + ")"
    elif script.type is NUMBER:
        return "Double.valueOf(" + text_type(script.expr) + ")"
    else:
        return script.expr