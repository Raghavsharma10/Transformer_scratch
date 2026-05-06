def unShare(sharedItem):
    """
    Remove all instances of this item from public or shared view.
    """
    sharedItem.store.query(Share, Share.sharedItem == sharedItem).deleteFromStore()