def result_headers(context, cl): 
    """
    Generates the list column headers.
    """
    for i, field_name in enumerate(cl.list_display):
        text, attr = label_for_field(field_name, cl.model, return_attr=True)
        if attr:
            # Potentially not sortable

            # if the field is the action checkbox: no sorting and special class
            if field_name == 'action_checkbox':
                yield {
                    "text": text,
                    "class_attrib": mark_safe(' class="action-checkbox-column"'),
                    "sortable": False,
                }
                continue

#            if other_fields like Edit, Visualize, etc:
                # Not sortable
#                yield {
#                    "text": text,
#                    "class_attrib": format_html(' class="column-{0}"', field_name),
#                    "sortable": False,
#                }
#                continue

        # OK, it is sortable if we got this far
        th_classes = ['sortable', 'column-{0}'.format(field_name)]
        ascending = None
        is_sorted = False
        # Is it currently being sorted on?
        if context.get('current_sort_field') == str(i + 1):
            is_sorted = True
            ascending = False
            th_classes.append('sorted descending')
        elif context.get('current_sort_field') == '-'+str(i + 1):
            is_sorted = True
            ascending = True
            th_classes.append('sorted ascending')

        if 'request' in context:
            url = context['request'].path
        else:
            url = "./"

### TODO: when start using action_checkbox use i instead of i + 1. This +1 is to correct enumerate index
        # builds url
        url += "?sort_by="

        if ascending is False:
            url += "-"
        url += str(i + 1)

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

        yield {
            "text": text,
            "url": url,
            "sortable": True,
            "sorted": is_sorted,
            "ascending": ascending,
            "class_attrib": format_html(' class="{0}"', ' '.join(th_classes)) if th_classes else '',
        }