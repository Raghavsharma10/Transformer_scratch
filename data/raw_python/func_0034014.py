def handle_special_cases(request, data, baseURI, meta):
    """Handle sepcial cases for returned values by the doAction function"""

    if request.method == 'OPTIONS':
        r = HttpResponse('')
        return r

    if data is None:
        return gen404(request, baseURI, 'data')

    if data.__class__.__name__ == 'PlugIt500':
        return gen500(request, baseURI)

    if data.__class__.__name__ == 'PlugItSpecialCode':
        r = HttpResponse('')
        r.status_code = data.code
        return r

    if data.__class__.__name__ == 'PlugItRedirect':
        url = data.url
        if not data.no_prefix:
            url = baseURI + url

        return HttpResponseRedirect(url)

    if data.__class__.__name__ == 'PlugItFile':
        response = HttpResponse(data.content, content_type=data.content_type)
        response['Content-Disposition'] = data.content_disposition

        return response

    if data.__class__.__name__ == 'PlugItNoTemplate':
        response = HttpResponse(data.content)
        return response

    if meta.get('json_only', None):  # Just send the json back
        # Return application/json if requested
        if 'HTTP_ACCEPT' in request.META and request.META['HTTP_ACCEPT'].find('json') != -1:
            return JsonResponse(data)

        # Return json data without html content type, since json was not
        # requiered
        result = json.dumps(data)
        return HttpResponse(result)

    if meta.get('xml_only', None):  # Just send the xml back
        return HttpResponse(data['xml'], content_type='application/xml')