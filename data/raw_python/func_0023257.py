def _last_stack_str():
    """Print stack trace from call that didn't originate from here"""
    stack = extract_stack()
    for s in stack[::-1]:
        if op.join('vispy', 'gloo', 'buffer.py') not in __file__:
            break
    return format_list([s])[0]