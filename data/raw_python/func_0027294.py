def list(self, request, *args, **kwargs):
        """
        To get a list of price list items, run **GET** against */api/merged-price-list-items/*
        as authenticated user.

        If service is not specified default price list items are displayed.
        Otherwise service specific price list items are displayed.
        In this case rendered object contains {"is_manually_input": true}

        In order to specify service pass query parameters:
        - service_type (Azure, OpenStack etc.)
        - service_uuid

        Example URL: http://example.com/api/merged-price-list-items/?service_type=Azure&service_uuid=cb658b491f3644a092dd223e894319be
        """
        return super(MergedPriceListItemViewSet, self).list(request, *args, **kwargs)