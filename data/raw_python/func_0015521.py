def add_bindings(self, g: Graph) -> "PrefixLibrary":
        """ Add bindings in the library to the graph

        :param g: graph to add prefixes to
        :return: PrefixLibrary object
        """
        for prefix, namespace in self:
            g.bind(prefix.lower(), namespace)
        return self