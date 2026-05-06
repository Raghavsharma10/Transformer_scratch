def __update_consistent_multicolors(self):
        """ Internally used method, that recalculates T-consistent / VT-consistent multicolors for current tree topology
        """
        v_t_consistent_multicolors = self.__get_v_tree_consistent_leaf_based_hashable_multicolors()

        hashed_vtree_consistent_leaves_multicolors = {mc.hashable_representation for mc in v_t_consistent_multicolors}
        self.vtree_consistent_multicolors_set = hashed_vtree_consistent_leaves_multicolors
        self.vtree_consistent_multicolors = [Multicolor(*hashed_multicolor) for hashed_multicolor in
                                             hashed_vtree_consistent_leaves_multicolors]
        result = []
        # T-consistent multicolors can be viewed as VT-consistent multicolors united with all of their complements
        full_multicolor = v_t_consistent_multicolors[0]
        for multicolor in v_t_consistent_multicolors:
            result.append(multicolor)
            result.append(full_multicolor - multicolor)

        hashed_tree_consistent_leaves_multicolors = {mc.hashable_representation for mc in result}
        self.tree_consistent_multicolors_set = hashed_tree_consistent_leaves_multicolors
        self.tree_consistent_multicolors = [Multicolor(*hashed_multicolor) for hashed_multicolor in
                                            hashed_tree_consistent_leaves_multicolors]
        self.multicolors_are_up_to_date = True