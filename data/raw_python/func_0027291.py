def update(self, request, *args, **kwargs):
        """
        Run **PATCH** request against */api/price-list-items/<uuid>/* to update price list item.
        Only item_type, key value and units can be updated.
        Only customer owner and staff can update price items.
        """
        return super(PriceListItemViewSet, self).update(request, *args, **kwargs)