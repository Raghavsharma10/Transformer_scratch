def upgradePrivateApplication3to4(old):
    """
    Upgrade L{PrivateApplication} from schema version 3 to schema version 4.

    Copy all existing attributes to the new version and use the
    L{PrivateApplication} to power up the item it is installed on for
    L{ITemplateNameResolver}.
    """
    new = old.upgradeVersion(
        PrivateApplication.typeName, 3, 4,
        preferredTheme=old.preferredTheme,
        privateKey=old.privateKey,
        website=old.website,
        customizedPublicPage=old.customizedPublicPage,
        authenticationApplication=old.authenticationApplication,
        preferenceAggregator=old.preferenceAggregator,
        defaultPreferenceCollection=old.defaultPreferenceCollection,
        searchAggregator=old.searchAggregator)
    # Almost certainly this would be more correctly expressed as
    # installedOn(new).powerUp(...), however the 2 to 3 upgrader failed to
    # translate the installedOn attribute to state which installedOn can
    # recognize, consequently installedOn(new) will return None for an item
    # which was created at schema version 2 or earlier.  It's not worth dealing
    # with this inconsistency, since PrivateApplication is always only
    # installed on its store. -exarkun
    new.store.powerUp(new, ITemplateNameResolver)
    return new