def execute(self, elem_list):
        """
        If condition, return a new elem_list provided by executing action.
        """
        if self.condition.is_true(elem_list):
            return self.action.act(elem_list)
        else:
            return elem_list