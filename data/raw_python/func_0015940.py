def get_queryset(self):
        '''Apply Datatables sort and search criterion to QuerySet'''
        qs = super(DatatablesView, self).get_queryset()
        # Perform global search
        qs = self.global_search(qs)
        # Perform column search
        qs = self.column_search(qs)
        # Return the ordered queryset
        return qs.order_by(*self.get_orders())