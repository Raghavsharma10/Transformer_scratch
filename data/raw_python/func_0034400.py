def __get_v_tree_consistent_leaf_based_hashable_multicolors(self):
        """ Internally used method, that recalculates VTree-consistent sets of leaves in the current tree """
        result = []
        nodes = deque([self.__root])
        while len(nodes) > 0:
            current_node = nodes.popleft()
            children = current_node.children
            nodes.extend(children)
            if not current_node.is_leaf():
                leaves = filter(lambda node: node.is_leaf(), current_node.get_descendants())
                result.append(Multicolor(*[self.__leaf_wrapper(leaf.name) for leaf in leaves]))
            else:
                result.append(Multicolor(self.__leaf_wrapper(current_node.name)))
        result.append(Multicolor())
        return result