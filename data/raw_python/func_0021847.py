def pypi_search(self):
        """
        Search PyPI by metadata keyword
        e.g. yolk -S name=yolk AND license=GPL

        @param spec: Cheese Shop search spec
        @type spec: list of strings

        spec examples:
          ["name=yolk"]
          ["license=GPL"]
          ["name=yolk", "AND", "license=GPL"]

        @returns: 0 on success or 1 if mal-formed search spec

        """
        spec = self.pkg_spec
        #Add remainging cli arguments to options.pypi_search
        search_arg = self.options.pypi_search
        spec.insert(0, search_arg.strip())

        (spec, operator) = self.parse_search_spec(spec)
        if not spec:
            return 1
        for pkg in self.pypi.search(spec, operator):
            if pkg['summary']:
                summary = pkg['summary'].encode('utf-8')
            else:
                summary = ""
            print("""%s (%s):
        %s
    """ % (pkg['name'].encode('utf-8'), pkg["version"],
                    summary))
        return 0