def setTabURLs(tabs, webTranslator):
    """
    Sets the C{linkURL} attribute on each L{Tab} instance
    in C{tabs} that does not already have it set

    @param tabs: sequence of L{Tab} instances
    @param webTranslator: L{xmantissa.ixmantissa.IWebTranslator}
                          implementor

    @return: None
    """

    for tab in tabs:
        if not tab.linkURL:
            tab.linkURL = webTranslator.linkTo(tab.storeID)
        setTabURLs(tab.children, webTranslator)