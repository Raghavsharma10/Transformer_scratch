def sort_headers(context, cl):
    """
    Displays the headers and data list together
    """
    headers = list(result_headers(context, cl))
    sorted_fields = False
    for h in headers:
        if h['sortable'] and h['sorted']:
            sorted_fields = True
    return {'cl': cl,
            'result_headers': headers,
            'sorted_fields': sorted_fields}