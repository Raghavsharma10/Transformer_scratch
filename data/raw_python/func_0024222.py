def _format_subtree(self, subtree):
        """Recursively format all subtrees."""
        subtree['children'] = list(subtree['children'].values())
        for child in subtree['children']:
            self._format_subtree(child)
        return subtree