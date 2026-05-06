def get_group_tree_root(self, page_size=1000):
        r"""Return the root group for this accounts' group tree

        This will return the root group for this tree but with all links
        between nodes (i.e. children starting from root) populated.

        Examples::

            # print the group hierarchy to stdout
            dc.devicecore.get_group_tree_root().print_subtree()

            # gather statistics about devices in each group including
            # the count from its subgroups (recursively)
            #
            # This also shows how you can go from a group reference to devices
            # for that particular group.
            stats = {}  # group -> devices count including children
            def count_nodes(group):
                count_for_this_node = \
                    len(list(dc.devicecore.get_devices(group_path == group.get_path())))
                subnode_count = 0
                for child in group.get_children():
                    subnode_count += count_nodes(child)
                total = count_for_this_node + subnode_count
                stats[group] = total
                return total
            count_nodes(dc.devicecore.get_group_tree_root())

        :param int page_size: The number of results to fetch in a
            single page.  In general, the default will suffice.
        :returns: The root group for this device cloud accounts group
            hierarchy.

        """

        # first pass, build mapping
        group_map = {}  # map id -> group
        page_size = validate_type(page_size, *six.integer_types)
        for group in self.get_groups(page_size=page_size):
            group_map[group.get_id()] = group

        # second pass, find root and populate list of children for each node
        root = None
        for group_id, group in group_map.items():
            if group.is_root():
                root = group
            else:
                parent = group_map[group.get_parent_id()]
                parent.add_child(group)
        return root