def set_value(self, iter, column, value):
        """Set the value of the child model"""

        # Delegate to child model
        iter = self.convert_iter_to_child_iter(iter)
        self.get_model().set_value(iter, column, value)