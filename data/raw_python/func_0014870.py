def sort_key_process(request, sort_key='sort'):
    """
        process sort-parameter value (for example, "-name")
        return:
            current_param - field for sorting ("name)
            current_reversed - revers flag (True)
    """
    current = request.GET.get(sort_key)
    current_reversed = False
    current_param = None
    if current:
        mo = re.match(r'^(-?)(\w+)$', current)    # exclude first "-" (if exist)
        if mo:
            current_reversed = mo.group(1) == '-'
            current_param = mo.group(2)

    return current_param, current_reversed