def link(self, selector):
        """
        Finds and returns a link ``<a>`` element (:class:`zombie.dom.DOMNode`).
        You can use a CSS selector or find a link by its text contents (case
        sensitive, but ignores leading/trailing spaces).

        :param selector: an optional string CSS selector
                        (http://zombie.labnotes.org/selectors) or inner text
        """
        element = self.client.create_element('browser.link', (selector,))
        return DOMNode(element, self)