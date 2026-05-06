def get(self, group, default=None):
        '''Get a group or parameter.

        Parameters
        ----------
        group : str
            If this string contains a period (.), then the part before the
            period will be used to retrieve a group, and the part after the
            period will be used to retrieve a parameter from that group. If this
            string does not contain a period, then just a group will be
            returned.
        default : any
            Return this value if the named group and parameter are not found.

        Returns
        -------
        value : :class:`Group` or :class:`Param`
            Either a group or parameter with the specified name(s). If neither
            is found, returns the default value.
        '''
        if isinstance(group, int):
            return self.groups.get(group, default)
        group = group.upper()
        param = None
        if '.' in group:
            group, param = group.split('.', 1)
        if ':' in group:
            group, param = group.split(':', 1)
        if group not in self.groups:
            return default
        group = self.groups[group]
        if param is not None:
            return group.get(param, default)
        return group