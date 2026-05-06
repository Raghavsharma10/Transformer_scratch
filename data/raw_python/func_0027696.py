def dependentItems(store, tableClass, comparisonFactory):
    """
    Collect all the items that should be deleted when an item or items
    of a particular item type are deleted.

    @param tableClass: An L{Item} subclass.

    @param comparison: A one-argument callable taking an attribute and
    returning an L{iaxiom.IComparison} describing the items to
    collect.

    @return: An iterable of items to delete.
    """
    for cascadingAttr in (_cascadingDeletes.get(tableClass, []) +
                          _cascadingDeletes.get(None, [])):
        for cascadedItem in store.query(cascadingAttr.type,
                                        comparisonFactory(cascadingAttr)):
            yield cascadedItem