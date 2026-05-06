def as_search_document_update(self, *, index, update_fields):
        """
        Return a partial update document based on which fields have been updated.

        If an object is saved with the `update_fields` argument passed
        through, then it is assumed that this is a 'partial update'. In
        this scenario we need a {property: value} dictionary containing
        just the fields we want to update.

        This method handles two possible update strategies - 'full' or 'partial'.
        The default 'full' strategy simply returns the value of `as_search_document`
        - thereby replacing the entire document each time. The 'partial' strategy is
        more intelligent - it will determine whether the fields passed are in the
        search document mapping, and return a partial update document that contains
        only those that are. In addition, if any field that _is_ included cannot
        be automatically serialized (e.g. a RelatedField object), then this method
        will raise a ValueError. In this scenario, you should override this method
        in your subclass.

        >>> def as_search_document_update(self, index, update_fields):
        ...     if 'user' in update_fields:
        ...         update_fields.remove('user')
        ...         doc = super().as_search_document_update(index, update_fields)
        ...         doc['user'] = self.user.get_full_name()
        ...         return doc
        ...     return super().as_search_document_update(index, update_fields)

        You may also wish to subclass this method to perform field-specific logic
        - in this example if only the timestamp is being saved, then ignore the
        update if the timestamp is later than a certain time.

        >>> def as_search_document_update(self, index, update_fields):
        ...     if update_fields == ['timestamp']:
        ...         if self.timestamp > today():
        ...            return {}
        ...     return super().as_search_document_update(index, update_fields)

        """
        if UPDATE_STRATEGY == UPDATE_STRATEGY_FULL:
            return self.as_search_document(index=index)

        if UPDATE_STRATEGY == UPDATE_STRATEGY_PARTIAL:
            # in partial mode we update the intersection of update_fields and
            # properties found in the mapping file.
            return {
                k: getattr(self, k)
                for k in self.clean_update_fields(
                    index=index, update_fields=update_fields
                )
            }