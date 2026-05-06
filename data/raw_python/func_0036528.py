def alphafilter(request, queryset, template):
    """
    Render the template with the filtered queryset
    """

    qs_filter = {}
    for key in list(request.GET.keys()):
        if '__istartswith' in key:
            qs_filter[str(key)] = request.GET[key]
            break

    return render_to_response(
        template,
        {'objects': queryset.filter(**qs_filter),
         'unfiltered_objects': queryset},
        context_instance=RequestContext(request)
    )