def applicationNavigation(ctx, translator, navigation):
    """
    Horizontal, primary-only navigation view.

    For the navigation element currently being viewed, copies of the
    I{selected-app-tab} and I{selected-tab-contents} patterns will be
    loaded from the tag.  For all other navigation elements, copies of the
    I{app-tab} and I{tab-contents} patterns will be loaded.

    For either case, the former pattern will have its I{name} slot filled
    with the name of the navigation element and its I{tab-contents} slot
    filled with the latter pattern.  The latter pattern will have its
    I{href} slot filled with a link to the corresponding navigation
    element.

    The I{tabs} slot on the tag will be filled with all the
    I{selected-app-tab} or I{app-tab} pattern copies.

    @type ctx: L{nevow.context.WebContext}
    @type translator: L{IWebTranslator} provider
    @type navigation: L{list} of L{Tab}

    @rtype: {nevow.stan.Tag}
    """
    setTabURLs(navigation, translator)
    selectedTab = getSelectedTab(navigation,
                                 url.URL.fromContext(ctx))

    getp = IQ(ctx.tag).onePattern
    tabs = []

    for tab in navigation:
        if tab == selectedTab or selectedTab in tab.children:
            p = 'selected-app-tab'
            contentp = 'selected-tab-contents'
        else:
            p = 'app-tab'
            contentp = 'tab-contents'

        childTabs = []
        for subtab in tab.children:
            try:
                subtabp = getp("subtab")
            except NodeNotFound:
                continue
            childTabs.append(
                dictFillSlots(subtabp, {
                        'name': subtab.name,
                        'href': subtab.linkURL,
                        'tab-contents': getp("subtab-contents")
                        }))
        tabs.append(dictFillSlots(
                getp(p),
                {'name': tab.name,
                 'tab-contents': getp(contentp).fillSlots(
                        'href', tab.linkURL),
                 'subtabs': childTabs}))

    ctx.tag.fillSlots('tabs', tabs)
    return ctx.tag