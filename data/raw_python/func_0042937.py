def __value_compare(self, target):
        """
        Comparing result based on expectation if arg_type is "VALUE"
        Args: Anything
        Return: Boolean
        """
        if self.expectation == "__ANY__":
            return True
        elif self.expectation == "__DEFINED__":
            return True if target is not None else False
        elif self.expectation == "__TYPE__":
            return True if type(target) == self.target_type else False #pylint:disable=unidiomatic-typecheck
        elif self.expectation == "__INSTANCE__":
            return True if isinstance(target, self.target_type.__class__) else False
        else:
            return True if target == self.expectation else False