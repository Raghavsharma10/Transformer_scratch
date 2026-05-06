def extract_literal(templatevar):
    """
    See if a template FilterExpression holds a literal value.

    :type templatevar: django.template.FilterExpression
    :rtype: bool|None
    """
    # FilterExpression contains another 'var' that either contains a Variable or SafeData object.
    if hasattr(templatevar, 'var'):
        templatevar = templatevar.var
        if isinstance(templatevar, SafeData):
            # Literal in FilterExpression, can return.
            return templatevar
        else:
            # Variable in FilterExpression, not going to work here.
            return None

    if templatevar[0] in ('"', "'") and templatevar[-1] in ('"', "'"):
        return templatevar[1:-1]
    else:
        return None