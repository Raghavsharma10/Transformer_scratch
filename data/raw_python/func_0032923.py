def linkTo(sharedProxyOrItem):
    """
    Generate the path part of a URL to link to a share item or its proxy.

    @param sharedProxy: a L{sharing.SharedProxy} or L{sharing.Share}

    @return: a URL object, which when converted to a string will look
        something like '/users/user@host/shareID'.

    @rtype: L{nevow.url.URL}

    @raise: L{RuntimeError} if the store that the C{sharedProxyOrItem} is
        stored in is not accessible via the web, for example due to the fact
        that the store has no L{LoginMethod} objects to indicate who it is
        owned by.
    """
    if isinstance(sharedProxyOrItem, sharing.SharedProxy):
        userStore = sharing.itemFromProxy(sharedProxyOrItem).store
    else:
        userStore = sharedProxyOrItem.store
    appStore = isAppStore(userStore)
    if appStore:
        # This code-path should be fixed by #2703; PublicWeb is deprecated.
        from xmantissa.publicweb import PublicWeb
        substore = userStore.parent.getItemByID(userStore.idInParent)
        pw = userStore.parent.findUnique(PublicWeb, PublicWeb.application == substore)
        path = [pw.prefixURL.encode('ascii')]
    else:
        for lm in userbase.getLoginMethods(userStore):
            if lm.internal:
                path = ['users', lm.localpart.encode('ascii')]
                break
        else:
            raise RuntimeError(
                "Shared item is in a user store with no"
                " internal username -- can't generate a link.")
    if (sharedProxyOrItem.shareID == getDefaultShareID(userStore)):
        shareID = sharedProxyOrItem.shareID
        path.append('')
    else:
        shareID = None
        path.append(sharedProxyOrItem.shareID)
    return _ShareURL(shareID, scheme='', netloc='', pathsegs=path)