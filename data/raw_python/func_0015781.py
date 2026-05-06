def get_es(self, default_builder=get_es):
        """Returns the Elasticsearch object to use.

        :arg default_builder: The function that takes a bunch of
            arguments and generates a elasticsearch Elasticsearch
            object.

        .. Note::

           If you desire special behavior regarding building the
           Elasticsearch object for this S, subclass S and override
           this method.

        """
        # .es() calls are incremental, so we go through them all and
        # update bits that are specified.
        args = {}
        for action, value in self.steps:
            if action == 'es':
                args.update(**value)

        # TODO: store the Elasticsearch on the S if we've already
        # created one since we don't need to do it multiple times.
        return default_builder(**args)