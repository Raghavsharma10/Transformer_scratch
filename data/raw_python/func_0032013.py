def sort_link(context, text, sort_field, visible_name=None):
    """Usage: {% sort_link "text" "field_name" %}
    Usage: {% sort_link "text" "field_name" "Visible name" %}
    """
    sorted_fields = False
    ascending = None
    class_attrib = 'sortable'
    orig_sort_field = sort_field
    if context.get('current_sort_field') == sort_field:
        sort_field = '-%s' % sort_field
        visible_name = '-%s' % (visible_name or orig_sort_field)
        sorted_fields = True
        ascending = False
        class_attrib += ' sorted descending'
    elif context.get('current_sort_field') == '-'+sort_field:
        visible_name = '%s' % (visible_name or orig_sort_field)
        sorted_fields = True
        ascending = True
        class_attrib += ' sorted ascending'

    if visible_name:
        if 'request' in context:
            request = context['request']
            request.session[visible_name] = sort_field

    # builds url
    if 'request' in context:
        url = context['request'].path
    else:
        url = "./"

    url += "?sort_by="
    if visible_name is None:
        url += sort_field
    else:
        url += visible_name

    if 'getsortvars' in context:
        extra_vars = context['getsortvars']
    else:
        if 'request' in context:
            request = context['request']
            getvars = request.GET.copy()
            if 'sort_by' in getvars:
                del getvars['sort_by']
            if len(getvars.keys()) > 0:
                context['getsortvars'] = "&%s" % getvars.urlencode()
            else:
                context['getsortvars'] = ''
            extra_vars = context['getsortvars']

    # append other vars to url
    url += extra_vars

    return {
        'text': text, 'url': url, 'ascending': ascending, 'sorted_fields': sorted_fields, 'class_attrib': class_attrib
    }