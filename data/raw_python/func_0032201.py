def mugshot1to2(old):
    """
    Upgrader for L{Mugshot} from version 1 to version 2, which sets the
    C{smallerBody} attribute to the path of a smaller mugshot image.
    """
    smallerBody = Mugshot.makeThumbnail(old.body.open(),
                                        old.person,
                                        old.type.split('/')[1],
                                        smaller=True)

    return old.upgradeVersion(Mugshot.typeName, 1, 2,
                              person=old.person,
                              type=old.type,
                              body=old.body,
                              smallerBody=smallerBody)