def startMenu(translator, navigation, tag):
    """
    Drop-down menu-style navigation view.

    For each primary navigation element available, a copy of the I{tab}
    pattern will be loaded from the tag.  It will have its I{href} slot
    filled with the URL for that navigation item.  It will have its I{name}
    slot filled with the user-visible name of the navigation element.  It
    will have its I{kids} slot filled with a list of secondary navigation
    for that element.

    For each secondary navigation element available beneath each primary
    navigation element, a copy of the I{subtabs} pattern will be loaded
    from the tag.  It will have its I{kids} slot filled with a self-similar
    structure.

    @type translator: L{IWebTranslator} provider
    @type navigation: L{list} of L{Tab}

    @rtype: {nevow.stan.Tag}
    """
    setTabURLs(navigation, translator)
    getp = IQ(tag).onePattern

    def fillSlots(tabs):
        for tab in tabs:
            if tab.children:
                kids = getp('subtabs').fillSlots('kids', fillSlots(tab.children))
            else:
                kids = ''

            yield dictFillSlots(getp('tab'), dict(href=tab.linkURL,
                                                  name=tab.name,
                                                  kids=kids))
    return tag.fillSlots('tabs', fillSlots(navigation))