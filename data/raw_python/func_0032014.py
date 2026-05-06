def auto_sort(parser, token):
    "usage: {% auto_sort queryset %}"
    try:
        tag_name, queryset = token.split_contents()
    except ValueError:
        raise template.TemplateSyntaxError("{0} tag requires a single argument".format(token.contents.split()[0]))
    return SortedQuerysetNode(queryset)