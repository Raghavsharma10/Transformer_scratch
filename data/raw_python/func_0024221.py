def _traverse_tree(tree, path):
        """Traverses the permission tree, returning the permission at given permission path."""
        path_steps = (step for step in path.split('.') if step != '')
        # Special handling for first step, because the first step isn't under 'objects'
        first_step = path_steps.next()
        subtree = tree[first_step]

        for step in path_steps:
            subtree = subtree['children'][step]
        return subtree