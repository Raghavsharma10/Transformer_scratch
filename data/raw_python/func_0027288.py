def list(self, request, *args, **kwargs):
        """
        To get a list of price estimates, run **GET** against */api/price-estimates/* as authenticated user.
        You can filter price estimates by scope type, scope URL, customer UUID.

        `scope_type` is generic type of object for which price estimate is calculated.
        Currently there are following types: customer, project, service, serviceprojectlink, resource.

        `date` parameter accepts list of dates. `start` and `end` parameters together specify date range.
        Each valid date should in format YYYY.MM

        You can specify GET parameter ?depth to show price estimate children. For example with ?depth=2 customer
        price estimate will shows its children - project and service and grandchildren - serviceprojectlink.
        """
        return super(PriceEstimateViewSet, self).list(request, *args, **kwargs)