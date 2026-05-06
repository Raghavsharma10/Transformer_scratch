def init_models(self) -> None:
        """Call method |Element.init_model| of all handle |Element| objects.

        We show, based the `LahnH` example project, that method
        |Element.init_model| prepares the |Model| objects of all elements,
        including building the required connections and updating the
        derived parameters:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import HydPy, pub, TestIO
        >>> with TestIO():
        ...     hp = HydPy('LahnH')
        ...     pub.timegrids = '1996-01-01', '1996-02-01', '1d'
        ...     hp.prepare_network()
        ...     hp.init_models()
        >>> hp.elements.land_dill.model.parameters.derived.dt
        dt(0.000833)

        Wrong control files result in error messages like the following:

        >>> with TestIO():
        ...     with open('LahnH/control/default/land_dill.py', 'a') as file_:
        ...         _ = file_.write('zonetype(-1)')
        ...     hp.init_models()   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: While trying to initialise the model object of element \
`land_dill`, the following error occurred: While trying to load the control \
file `...land_dill.py`, the following error occurred: At least one value of \
parameter `zonetype` of element `?` is not valid.

        By default, missing control files result in exceptions:

        >>> del hp.elements.land_dill.model
        >>> import os
        >>> with TestIO():
        ...     os.remove('LahnH/control/default/land_dill.py')
        ...     hp.init_models()   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: While trying to initialise the model object of \
element `land_dill`, the following error occurred: While trying to load the \
control file `...land_dill.py`, the following error occurred: ...
        >>> hasattr(hp.elements.land_dill, 'model')
        False

        When building new, still incomplete *HydPy* projects, this behaviour
        can be annoying.  After setting the option
        |Options.warnmissingcontrolfile| to |False|, missing control files
        only result in a warning:

        >>> with TestIO():
        ...     with pub.options.warnmissingcontrolfile(True):
        ...         hp.init_models()
        Traceback (most recent call last):
        ...
        UserWarning: Due to a missing or no accessible control file, \
no model could be initialised for element `land_dill`
        >>> hasattr(hp.elements.land_dill, 'model')
        False
        """
        try:
            for element in printtools.progressbar(self):
                element.init_model(clear_registry=False)
        finally:
            hydpy.pub.controlmanager.clear_registry()