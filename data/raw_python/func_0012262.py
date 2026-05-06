async def _has_id(self, *args, **kwds):
        """
            Equality checks are overwitten to perform the actual check in a
            semantic way.
        """
        # if there is only one positional argument
        if len(args) == 1:
            # parse the appropriate query
            result = await parse_string(
                self._query,
                self.service.object_resolver,
                self.service.connection_resolver,
                self.service.mutation_resolver,
                obey_auth=False
            )
            # go to the bottom of the result for the list of matching ids
            return self._find_id(result['data'], args[0])
        # otherwise
        else:
            # treat the attribute like a normal filter
            return self._has_id(**kwds)