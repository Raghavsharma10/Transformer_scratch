def get_pretty_format(self, include_id=True, max_name_length=0,
                          abbreviate=True):
        """Returns a nicely formatted string with the GO term information.

        Parameters
        ----------
        include_id: bool, optional
            Include the GO term ID.
        max_name_length: int, optional
            Truncate the formatted string so that its total length does not
            exceed this value.
        abbreviate: bool, optional
            Do not use abberviations (see ``_abbrev``) to shorten the GO term
            name.

        Returns
        -------
        str
            The formatted string.
        """
        name = self.name
        if abbreviate:
            for abb in self._abbrev:
                name = re.sub(abb[0], abb[1], name)
        if 3 <= max_name_length < len(name):
            name = name[:(max_name_length-3)] + '...'
        if include_id:
            return "%s: %s (%s)" % (self.domain_short, name, self.id)
        else:
            return "%s: %s" % (self.domain_short, name)