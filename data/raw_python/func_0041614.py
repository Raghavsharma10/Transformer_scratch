def add_link(self, targets, weight):
        """
        Add link(s) pointing to ``targets``.

        If a link already exists pointing to a target, just add ``weight``
        to that link's weight

        Args:
            targets (Node or list[Node]): node or nodes to link to
            weight (int or float): weight for the new link(s)

        Returns: None

        Example:
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_2, 1)
            >>> new_link = node_1.link_list[0]
            >>> print(new_link)
            node.Link instance pointing to node with value "Two" with weight 1
        """
        # Generalize targets to a list to simplify code
        if not isinstance(targets, list):
            target_list = [targets]
        else:
            target_list = targets

        for target in target_list:
            # Check to see if self already has a link to target
            for existing_link in self.link_list:
                if existing_link.target == target:
                    existing_link.weight += weight
                    break
            else:
                self.link_list.append(Link(target, weight))