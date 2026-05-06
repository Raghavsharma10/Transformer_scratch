def upgradePrivateApplication4to5(old):
    """
    Install the newly required powerup.
    """
    new = old.upgradeVersion(
        PrivateApplication.typeName, 4, 5,
        preferredTheme=old.preferredTheme,
        privateKey=old.privateKey,
        website=old.website,
        customizedPublicPage=old.customizedPublicPage,
        authenticationApplication=old.authenticationApplication,
        preferenceAggregator=old.preferenceAggregator,
        defaultPreferenceCollection=old.defaultPreferenceCollection,
        searchAggregator=old.searchAggregator)
    new.store.powerUp(new, IWebViewer)
    return new