def get_queryset(self):
        """
        Returns queryset instance.

        :rtype: django.db.models.query.QuerySet.
        """
        queryset    = super(IndexView, self).get_queryset()
        search_form = self.get_search_form()

        if search_form.is_valid():
            query_str   = search_form.cleaned_data.get('q', '').strip()
            queryset    = self.model.objects.search(query_str)

        return queryset