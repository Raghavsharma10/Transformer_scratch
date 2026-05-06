def create_tree(self, tree, base_tree=''):
        """Create a tree on this repository.

        :param list tree: (required), specifies the tree structure.
            Format: [{'path': 'path/file', 'mode':
            'filemode', 'type': 'blob or tree', 'sha': '44bfc6d...'}]
        :param str base_tree: (optional), SHA1 of the tree you want
            to update with new data
        :returns: :class:`Tree <github3.git.Tree>` if successful, else None
        """
        json = None
        if tree and isinstance(tree, list):
            data = {'tree': tree, 'base_tree': base_tree}
            url = self._build_url('git', 'trees', base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
        return Tree(json) if json else None