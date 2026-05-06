def retrieve_list(self, session, filters, *args, **kwargs):
        """
        Retrieves a list of the model for this manager.
        It is restricted by the filters provided.

        :param Session session: The SQLAlchemy session to use
        :param dict filters: The filters to restrict the returned
            models on
        :return: A tuple of the list of dictionary representation
            of the models and the dictionary of meta data
        :rtype: list, dict
        """
        query = self.queryset(session)
        translator = IntegerField('tmp')
        pagination_count = translator.translate(
            filters.pop(self.pagination_count_query_arg, self.paginate_by)
        )
        pagination_pk = translator.translate(
            filters.pop(self.pagination_pk_query_arg, 1)
        )
        pagination_pk -= 1  # logic works zero based. Pagination shouldn't be though

        query = query.filter_by(**filters)

        if pagination_pk:
            query = query.offset(pagination_pk * pagination_count)
        if pagination_count:
            query = query.limit(pagination_count + 1)

        count = query.count()
        next_link = None
        previous_link = None
        if count > pagination_count:
            next_link = {self.pagination_pk_query_arg: pagination_pk + 2,
                         self.pagination_count_query_arg: pagination_count}
        if pagination_pk > 0:
            previous_link = {self.pagination_pk_query_arg: pagination_pk,
                             self.pagination_count_query_arg: pagination_count}

        field_dict = self.dot_field_list_to_dict(self.list_fields)
        props = self.serialize_model(query[:pagination_count], field_dict=field_dict)
        meta = dict(links=dict(next=next_link, previous=previous_link))
        return props, meta