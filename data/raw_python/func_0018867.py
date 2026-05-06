def alternative_initvalue(self) -> Union[bool, int, float]:
        """A user-defined value to be used instead of the value of class
        constant `INIT`.

        See the main documentation on class |SolverParameter| for more
        information.
        """
        if self._alternative_initvalue is None:
            raise AttributeError(
                f'No alternative initial value for solver parameter '
                f'{objecttools.elementphrase(self)} has been defined so far.')
        else:
            return self._alternative_initvalue