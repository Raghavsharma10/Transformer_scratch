def list(self, request, *args, **kwargs):
        """
        To get a list of price list items, run **GET** against */api/price-list-items/* as an authenticated user.
        """
        return super(PriceListItemViewSet, self).list(request, *args, **kwargs)