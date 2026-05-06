def get_context_data(self, **kwargs):
        """
        Add in what fields are linkable
        """
        context = super(SmartListView, self).get_context_data(**kwargs)

        # our linkable fields
        self.link_fields = self.derive_link_fields(context)

        # stuff it all in our context
        context['link_fields'] = self.link_fields

        # our search term if any
        if 'search' in self.request.GET:
            context['search'] = self.request.GET['search']

        # our ordering field if any
        order = self.derive_ordering()
        if order:
            if order[0] == '-':
                context['order'] = order[1:]
                context['order_asc'] = False
            else:
                context['order'] = order
                context['order_asc'] = True

        return context