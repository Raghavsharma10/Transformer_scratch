def upgradeCatalog1to2(oldCatalog):
    """
    Create _TagName instances which version 2 of Catalog automatically creates
    for use in determining the tagNames result, but which version 1 of Catalog
    did not create.
    """
    newCatalog = oldCatalog.upgradeVersion('tag_catalog', 1, 2,
                                           tagCount=oldCatalog.tagCount)
    tags = newCatalog.store.query(Tag, Tag.catalog == newCatalog)
    tagNames = tags.getColumn("name").distinct()
    for t in tagNames:
        _TagName(store=newCatalog.store, catalog=newCatalog, name=t)
    return newCatalog