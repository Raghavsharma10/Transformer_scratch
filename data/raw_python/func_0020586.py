def create_element(self, method, args=None):
        """
        Evaluate a browser method and CSS selector against the document
        (or an optional context DOMNode) and return a single
        :class:`zombie.dom.DOMNode` object, e.g.,

        browser._node('query', 'body > div')

        ...roughly translates to the following Javascript...

        browser.query('body > div')

        :param method: the method (e.g., query) to call on the browser
        :param selector: a string CSS selector
                        (http://zombie.labnotes.org/selectors)
        :param context: an (optional) instance of :class:`zombie.dom.DOMNode`
        """
        if args is None:
            arguments = ''
        else:
            arguments = "(%s)" % encode_args(args)
        js = """
            create_element(ELEMENTS, %(method)s%(args)s);
        """ % {
            'method': method,
            'args': arguments
        }

        index = self.json(js)
        if index is None:
            return None

        return Element(index)