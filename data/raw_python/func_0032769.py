def upgradeShare1to2(oldShare):
    "Upgrader from Share version 1 to version 2."
    sharedInterfaces = []
    attrs = set(oldShare.sharedAttributeNames.split(u','))
    for iface in implementedBy(oldShare.sharedItem.__class__):
        if set(iface) == attrs or attrs == set('*'):
            sharedInterfaces.append(iface)

    newShare = oldShare.upgradeVersion('sharing_share', 1, 2,
                                       shareID=oldShare.shareID,
                                       sharedItem=oldShare.sharedItem,
                                       sharedTo=oldShare.sharedTo,
                                       sharedInterfaces=sharedInterfaces)
    return newShare