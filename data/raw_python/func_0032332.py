def get_text_node(self, tree, xpath_str):
        """Return a text node from given XML tree given an lxml XPath."""
        try:
            text = tree.xpath(xpath_str, namespaces=self.namespaces)[0].text
            return text_type(text) if text else ''
        except IndexError:  # pragma: nocover
            return ''