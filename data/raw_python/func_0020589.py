def query(self, selector, context=None):
        """
        Evaluate a CSS selector against the document (or an optional context
        :class:`zombie.dom.DOMNode`) and return a single
        :class:`zombie.dom.DOMNode` object.

        :param selector: a string CSS selector
                        (http://zombie.labnotes.org/selectors)
        :param context: an (optional) instance of :class:`zombie.dom.DOMNode`
        """
        element = self.client.create_element(
            'browser.query', (selector, context))
        return DOMNode.factory(element, self)