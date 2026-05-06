def get_template(request, query, meta, proxyMode):
    """Return (if needed) the template to use"""

    templateContent = None

    if not proxyMode:

        templateContent = plugIt.getTemplate(query, meta)

        if not templateContent:
            return (None, gen404(request, baseURI, 'template'))

    return (templateContent, None)