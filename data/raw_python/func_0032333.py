def get_subtree(self, tree, xpath_str):
        """Return a subtree given an lxml XPath."""
        return tree.xpath(xpath_str, namespaces=self.namespaces)