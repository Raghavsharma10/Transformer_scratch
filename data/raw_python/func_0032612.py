def _reorderForPreference(themeList, preferredThemeName):
    """
    Re-order the input themeList according to the preferred theme.

    Returns None.
    """
    for theme in themeList:
        if preferredThemeName == theme.themeName:
            themeList.remove(theme)
            themeList.insert(0, theme)
            return