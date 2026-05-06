def _render(request, data, encrypted, format=None):
    """
    Render the data to Geckoboard. If the `format` parameter is passed
    to the widget it defines the output format. Otherwise the output
    format is based on the `format` request parameter.

    A `format` paramater of ``json`` or ``2`` renders JSON output, any
    other value renders XML.
    """
    if not format:
        format = request.POST.get('format', '')
    if not format:
        format = request.GET.get('format', '')
    if format == 'json' or format == '2':
        return _render_json(data, encrypted)
    else:
        return _render_xml(data, encrypted)