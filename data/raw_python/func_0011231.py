def replacement_fields_from_context(context):
    """Convert context replacement fields

    Example:
        BE_KEY=value -> {"key": "value}

    Arguments:
        context (dict): The current context

    """

    return dict((k[3:].lower(), context[k])
                for k in context if k.startswith("BE_"))