def _validate_entities(self, stages):
        """
        Purpose: Validate whether the argument 'stages' is of list of Stage objects

        :argument: list of Stage objects
        """
        if not stages:
            raise TypeError(expected_type=Stage, actual_type=type(stages))

        if not isinstance(stages, list):
            stages = [stages]

        for value in stages:
            if not isinstance(value, Stage):
                raise TypeError(expected_type=Stage, actual_type=type(value))

        return stages