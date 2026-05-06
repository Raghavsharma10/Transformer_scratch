def parse_search_spec(self, spec):
        """
        Parse search args and return spec dict for PyPI
        * Owwww, my eyes!. Re-write this.

        @param spec: Cheese Shop package search spec
                     e.g.
                     name=Cheetah
                     license=ZPL
                     license=ZPL AND name=Cheetah
        @type spec: string

        @returns:  tuple with spec and operator
        """

        usage = \
            """You can search PyPI by the following:
     name
     version
     author
     author_email
     maintainer
     maintainer_email
     home_page
     license
     summary
     description
     keywords
     platform
     download_url

     e.g. yolk -S name=Cheetah
          yolk -S name=yolk AND license=PSF
          """

        if not spec:
            self.logger.error(usage)
            return (None, None)

        try:
            spec = (" ").join(spec)
            operator = 'AND'
            first = second = ""
            if " AND " in spec:
                (first, second) = spec.split('AND')
            elif " OR " in spec:
                (first, second) = spec.split('OR')
                operator = 'OR'
            else:
                first = spec
            (key1, term1) = first.split('=')
            key1 = key1.strip()
            if second:
                (key2, term2) = second.split('=')
                key2 = key2.strip()

            spec = {}
            spec[key1] = term1
            if second:
                spec[key2] = term2
        except:
            self.logger.error(usage)
            spec = operator = None
        return (spec, operator)