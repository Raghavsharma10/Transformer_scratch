def get_kwargs(**kwargs):
        """This method should be used in query functions where user can query on any number of fields

            >>> def get_instances(entity_id=NOTSET, my_field=NOTSET):
            >>>     kwargs = CoyoteDb.get_kwargs(entity_id=entity_id, my_field=my_field)
        """
        d = dict()
        for k, v in kwargs.iteritems():
            if v is not NOTSET:
                d[k] = v
        return d