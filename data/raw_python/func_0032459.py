def getSelectedTab(tabs, forURL):
    """
    Returns the tab that should be selected when the current
    resource lives at C{forURL}.  Call me after L{setTabURLs}

    @param tabs: sequence of L{Tab} instances
    @param forURL: L{nevow.url.URL}

    @return: L{Tab} instance
    """

    flatTabs = []

    def flatten(tabs):
        for t in tabs:
            flatTabs.append(t)
            flatten(t.children)

    flatten(tabs)
    forURL = '/' + forURL.path

    for t in flatTabs:
        if forURL == t.linkURL:
            return t

    flatTabs.sort(key=lambda t: len(t.linkURL), reverse=True)

    for t in flatTabs:
        if not t.linkURL.endswith('/'):
            linkURL = t.linkURL + '/'
        else:
            linkURL = t.linkURL

        if forURL.startswith(linkURL):
            return t