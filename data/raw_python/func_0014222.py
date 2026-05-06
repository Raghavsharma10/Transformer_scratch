def _toggle_autoescape(context, escape_on=True):
    '''
    Internal method to toggle autoescaping on or off. This function
    needs access to the caller, so the calling method must be
    decorated with @supports_caller.
    '''
    previous = is_autoescape(context)
    setattr(context.caller_stack, AUTOESCAPE_KEY, escape_on)
    try:
        context['caller'].body()
    finally:
        setattr(context.caller_stack, AUTOESCAPE_KEY, previous)