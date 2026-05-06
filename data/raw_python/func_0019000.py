def plot_stateseries(
            self, names: Optional[Iterable[str]] = None,
            average: bool = False, **kwargs: Any) \
            -> None:
        """Plot the `state` series of the handled model.

        See the documentation on method |Element.plot_inputseries| for
        additional information.
        """
        self.__plot(self.model.sequences.states, names, average, kwargs)