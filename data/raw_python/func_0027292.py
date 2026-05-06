def destroy(self, request, *args, **kwargs):
        """
        Run **DELETE** request against */api/price-list-items/<uuid>/* to delete price list item.
        Only customer owner and staff can delete price items.
        """
        return super(PriceListItemViewSet, self).destroy(request, *args, **kwargs)