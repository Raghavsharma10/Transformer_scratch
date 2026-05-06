def create(self, request, *args, **kwargs):
        """
        Run **POST** request against */api/price-list-items/* to create new price list item.
        Customer owner and staff can create price items.

        Example of request:

        .. code-block:: http

            POST /api/price-list-items/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "units": "per month",
                "value": 100,
                "service": "http://example.com/api/oracle/d4060812ca5d4de390e0d7a5062d99f6/",
                "default_price_list_item": "http://example.com/api/default-price-list-items/349d11e28f634f48866089e41c6f71f1/"
            }
        """
        return super(PriceListItemViewSet, self).create(request, *args, **kwargs)