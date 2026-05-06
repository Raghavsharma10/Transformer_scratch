def navigation_info(request):
    '''Expose whether to display the navigation header and footer'''
    if request.GET.get('wafer_hide_navigation') == "1":
        nav_class = "wafer-invisible"
    else:
        nav_class = "wafer-visible"
    context = {
        'WAFER_NAVIGATION_VISIBILITY': nav_class,
    }
    return context