def conditions(self) -> Dict[str, Dict[str, Union[float, numpy.ndarray]]]:
        """Nested dictionary containing the values of all condition
        sequences.

        See the documentation on property |HydPy.conditions| for further
        information.
        """
        conditions = {}
        for subname in NAMES_CONDITIONSEQUENCES:
            subseqs = getattr(self, subname, ())
            subconditions = {seq.name: copy.deepcopy(seq.values)
                             for seq in subseqs}
            if subconditions:
                conditions[subname] = subconditions
        return conditions