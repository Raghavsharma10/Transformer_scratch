def no_constraints(cls, callback):
        """
        Runs a callback with constraints disabled on the relation.
        """
        cls._constraints = False

        results = callback()

        cls._constraints = True

        return results