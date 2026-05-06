def update_text(self, mapping):
        """Iterate over nodes, replace text with mapping"""
        found = False
        for node in self._page.iter("*"):
            if node.text or node.tail:
                for old, new in mapping.items():
                    if node.text and old in node.text:
                        node.text = node.text.replace(old, new)
                        found = True
                    if node.tail and old in node.tail:
                        node.tail = node.tail.replace(old, new)
                        found = True
        if not found:
            raise KeyError("Updating text failed with mapping:{}".format(mapping))