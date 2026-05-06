def in_search_queryset(self, instance_id, index="_all"):
        """
        Return True if an object is part of the search index queryset.

        Sometimes it's useful to know if an object _should_ be indexed. If
        an object is saved, how do you know if you should push that change
        to the search index? The simplest (albeit not most efficient) way
        is to check if it appears in the underlying search queryset.

        NB this method doesn't evaluate the entire dataset, it chains an
        additional queryset filter expression on the end. That's why it's
        important that the `get_search_queryset` method returns a queryset.

        Args:
            instance_id: the id of model object that we are looking for.

        Kwargs:
            index: string, the name of the index in which to check.
                Defaults to '_all'.

        """
        return self.get_search_queryset(index=index).filter(pk=instance_id).exists()