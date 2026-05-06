def create_elements(self, method, args=[]):
        """
        Execute a browser method that will return a list of elements.

        Returns a list of the element indexes
        """
        args = encode_args(args)

        js = """
            create_elements(ELEMENTS, %(method)s(%(args)s))
        """ % {
            'method': method,
            'args': args,
        }

        indexes = self.json(js)
        return map(Element, indexes)