def _textParameterToView(parameter):
    """
    Return a L{TextParameterView} adapter for C{TEXT_INPUT}, C{PASSWORD_INPUT},
    and C{FORM_INPUT} L{Parameter} instances.
    """
    if parameter.type == TEXT_INPUT:
        return TextParameterView(parameter)
    if parameter.type == PASSWORD_INPUT:
        return PasswordParameterView(parameter)
    if parameter.type == FORM_INPUT:
        return FormInputParameterView(parameter)
    return None