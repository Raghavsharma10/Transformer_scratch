def extract_literal_bool(templatevar):
    """
    See if a template FilterExpression holds a literal boolean value.

    :type templatevar: django.template.FilterExpression
    :rtype: bool|None
    """
    # FilterExpression contains another 'var' that either contains a Variable or SafeData object.
    if hasattr(templatevar, 'var'):
        templatevar = templatevar.var
        if isinstance(templatevar, SafeData):
            # Literal in FilterExpression, can return.
            return is_true(templatevar)
        else:
            # Variable in FilterExpression, not going to work here.
            return None

    return is_true(templatevar)