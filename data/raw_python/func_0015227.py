def get_subassistant_tree(self):
        """Returns a tree-like structure representing the assistant hierarchy going down
        from this assistant to leaf assistants.

        For example: [(<This Assistant>,
                       [(<Subassistant 1>, [...]),
                        (<Subassistant 2>, [...])]
                      )]
        Returns:
            a tree-like structure (see above) representing assistant hierarchy going down
            from this assistant to leaf assistants
        """
        if '_tree' not in dir(self):
            subassistant_tree = []
            subassistants = self.get_subassistants()
            for subassistant in subassistants:
                subassistant_tree.append(subassistant.get_subassistant_tree())
            self._tree = (self, subassistant_tree)
        return self._tree