def initialize(self, name=None, dbname=None, base=None, generator=None,
                   case=None, namespaces=None):

        self.name = none_or(name, str)
        """
        The name of the site. : str | `None`
        """

        self.dbname = none_or(dbname, str)
        """
        The dbname of the site. : str | `None`
        """

        self.base = none_or(base, str)
        """
        TODO: ??? : str | `None`
        """

        self.generator = none_or(generator, str)
        """
        TODO: ??? : str | `None`
        """

        self.case = none_or(case, str)
        """
        TODO: ??? : str | `None`
        """

        self.namespaces = none_or(namespaces, list)
        """
        A list of :class:`mwtypes.Namespace` | `None`
        """