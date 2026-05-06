def links(tself, group=None):
    '''Returns the HTML for the given provider group (or all groups if None)'''
    pr = ProviderRun(tself, group)
    pr.run()
    return mark_safe(pr.getvalue())