def conditions(self) -> \
            Dict[str, Dict[str, Dict[str, Union[float, numpy.ndarray]]]]:
        """A nested dictionary containing the values of all
        |ConditionSequence| objects of all currently handled models.

        See the documentation on property |HydPy.conditions| for further
        information.
        """
        return {element.name: element.model.sequences.conditions
                for element in self}