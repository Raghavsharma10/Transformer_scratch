def plot_inputseries(
            self, names: Optional[Iterable[str]] = None,
            average: bool = False, **kwargs: Any) \
            -> None:
        """Plot (the selected) |InputSequence| |IOSequence.series| values.

        We demonstrate the functionalities of method |Element.plot_inputseries|
        based on the `Lahn` example project:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, _, _ = prepare_full_example_2(lastdate='1997-01-01')

        Without any arguments, |Element.plot_inputseries| prints the
        time series of all input sequences handled by its |Model| object
        directly to the screen (in the given example, |hland_inputs.P|,
        |hland_inputs.T|, |hland_inputs.TN|, and |hland_inputs.EPN| of
        application model |hland_v1|):

        >>> land = hp.elements.land_dill
        >>> land.plot_inputseries()

        You can use the `pyplot` API of `matplotlib` to modify the figure
        or to save it to disk (or print it to the screen, in case the
        interactive mode of `matplotlib` is disabled):

        >>> from matplotlib import pyplot
        >>> from hydpy.docs import figs
        >>> pyplot.savefig(figs.__path__[0] + '/Element_plot_inputseries.png')
        >>> pyplot.close()

        .. image:: Element_plot_inputseries.png

        Methods |Element.plot_fluxseries| and |Element.plot_stateseries|
        work in the same manner.  Before applying them, one has at first
        to calculate the time series of the |FluxSequence| and
        |StateSequence| objects:

        >>> hp.doit()

        All three methods allow to select certain sequences by passing their
        names (here, flux sequences |hland_fluxes.Q0| and |hland_fluxes.Q1|
        of |hland_v1|). Additionally, you can pass the keyword arguments
        supported by `matplotlib` for modifying the line style:

        >>> land.plot_fluxseries(['q0', 'q1'], linewidth=2)

        >>> pyplot.savefig(figs.__path__[0] + '/Element_plot_fluxseries.png')
        >>> pyplot.close()

        .. image:: Element_plot_fluxseries.png

        For 1-dimensional |IOSequence| objects, all three methods plot the
        individual time series in the same colour (here, from the state
        sequences |hland_states.SP| and |hland_states.WC| of |hland_v1|):

        >>> land.plot_stateseries(['sp', 'wc'])

        >>> pyplot.savefig(figs.__path__[0] + '/Element_plot_stateseries1.png')
        >>> pyplot.close()

        .. image:: Element_plot_stateseries1.png

        Alternatively, you can print the averaged time series through
        passing |True| to the method `average` argument (demonstrated
        for the state sequence |hland_states.SM|):

        >>> land.plot_stateseries(['sm'], color='grey')
        >>> land.plot_stateseries(
        ...     ['sm'], average=True, color='black', linewidth=3)

        >>> pyplot.savefig(figs.__path__[0] + '/Element_plot_stateseries2.png')
        >>> pyplot.close()

        .. image:: Element_plot_stateseries2.png
        """
        self.__plot(self.model.sequences.inputs, names, average, kwargs)