def _collect_leaf_paths(self, node, path_parts, leaf_paths):
        """
        Collect paths of leaf nodes.

        :param node:
            Starting node. Type is dict.

            Key is child node's path part. Value is child node.

        :param path_parts:
            The starting node's path parts. Type is tuple.

        :param leaf_paths:
            Leaf path list.

        :return:
            None.
        """
        # If the node is leaf node
        if not node:
            # Get node path
            node_path = '/'.join(path_parts)

            # Add to list
            leaf_paths.add(node_path)

        # If the node is not leaf node
        else:
            # For each child node
            for child_path_part, child_node in node.items():
                # Get the child node's path parts
                child_path_part_s = path_parts + (child_path_part,)

                # Visit the child node
                self._collect_leaf_paths(
                    node=child_node,
                    path_parts=child_path_part_s,
                    leaf_paths=leaf_paths,
                )