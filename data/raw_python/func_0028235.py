def _find_short_paths(self, paths):
        """
        Find short paths of given paths.

        E.g. if both `/home` and `/home/aoik` exist, only keep `/home`.

        :param paths:
            Paths.

        :return:
            Set of short paths.
        """
        # Split each path to parts.
        # E.g. '/home/aoik' to ['', 'home', 'aoik']
        path_parts_s = [path.split(os.path.sep) for path in paths]

        # Root node
        root_node = {}

        # Sort these path parts by length, with the longest being the first.
        #
        # Longer paths appear first so that their extra parts are discarded
        # when a shorter path is found at 5TQ8L.
        #
        # Then for each path's parts.
        for parts in sorted(path_parts_s, key=len, reverse=True):
            # Start from the root node
            node = root_node

            # For each part of the path
            for part in parts:
                # Create node of the path
                node = node.setdefault(part, {})

            # 5TQ8L
            # Clear the last path part's node's child nodes.
            #
            # This aims to keep only the shortest path that needs be watched.
            #
            node.clear()

        # Short paths
        short_path_s = set()

        # Collect leaf paths
        self._collect_leaf_paths(
            node=root_node,
            path_parts=(),
            leaf_paths=short_path_s,
        )

        # Return short paths
        return short_path_s