def order_queryset(self, queryset):
        """
        Orders the passed in queryset, returning a new queryset in response.  By default uses the _order query
        parameter.
        """
        order = self.derive_ordering()

        # if we get our order from the request
        # make sure it is a valid field in the list
        if '_order' in self.request.GET:
            if order.lstrip('-') not in self.derive_fields():
                order = None

        if order:
            # if our order is a single string, convert to a simple list
            if isinstance(order, str):
                order = (order,)

            queryset = queryset.order_by(*order)

        return queryset