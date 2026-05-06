def include_fields(self, *args):
        r"""
            Include fields is the fields that you want to be returned when
            searching. These are in addition to the fields that are always
            included below.

            :param args: items passed in will be turned into a list
            :returns: :class:`Search`

            >>> bugzilla.search_for.include_fields("flags")

            The following fields are always included in search:
                'version', 'id', 'summary', 'status', 'op_sys',
                'resolution', 'product', 'component', 'platform'
        """
        for arg in args:
            self._includefields.append(arg)
        return self