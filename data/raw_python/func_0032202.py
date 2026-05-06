def mugshot2to3(old):
    """
    Upgrader for L{Mugshot} from version 2 to version 3, which re-thumbnails
    the mugshot to take into account the new value of L{Mugshot.smallerSize}.
    """
    new = old.upgradeVersion(Mugshot.typeName, 2, 3,
                             person=old.person,
                             type=old.type,
                             body=old.body,
                             smallerBody=old.smallerBody)
    new.smallerBody = new.makeThumbnail(
        new.body.open(), new.person, new.type[len('image/'):], smaller=True)
    return new