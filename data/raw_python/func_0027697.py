def allowDeletion(store, tableClass, comparisonFactory):
    """
    Returns a C{bool} indicating whether deletion of an item or items of a
    particular item type should be allowed to proceed.

    @param tableClass: An L{Item} subclass.

    @param comparison: A one-argument callable taking an attribute and
    returning an L{iaxiom.IComparison} describing the items to
    collect.

    @return: A C{bool} indicating whether deletion should be allowed.
    """
    for cascadingAttr in (_disallows.get(tableClass, []) +
                          _disallows.get(None, [])):
        for cascadedItem in store.query(cascadingAttr.type,
                                        comparisonFactory(cascadingAttr),
                                        limit=1):
            return False

    return True