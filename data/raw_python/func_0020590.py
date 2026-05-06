def queryAll(self, selector, context=None):
        """
        Evaluate a CSS selector against the document (or an optional context
        :class:`zombie.dom.DOMNode`) and return a list of
        :class:`zombie.dom.DOMNode` objects.

        :param selector: a string CSS selector
                        (http://zombie.labnotes.org/selectors)
        :param context: an (optional) instance of :class:`zombie.dom.DOMNode`
        """
        elements = self.client.create_elements(
            'browser.queryAll', (selector, context))
        return [DOMNode(e, self) for e in elements]