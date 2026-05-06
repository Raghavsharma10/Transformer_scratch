def model(self) -> 'modeltools.Model':
        """The |Model| object handled by the actual |Element| object.

        Directly after their initialisation, elements do not know
        which model they require:

        >>> from hydpy import Element
        >>> hland = Element('hland', outlets='outlet')
        >>> hland.model
        Traceback (most recent call last):
        ...
        AttributeError: The model object of element `hland` has been \
requested but not been prepared so far.

        During scripting and when working interactively in the Python
        shell, it is often convenient to assign a |model| directly.

        >>> from hydpy.models.hland_v1 import *
        >>> parameterstep('1d')
        >>> hland.model = model
        >>> hland.model.name
        'hland_v1'

        >>> del hland.model
        >>> hasattr(hland, 'model')
        False

        For the "usual" approach to prepare models, please see the method
        |Element.init_model|.

        The following examples show that assigning |Model| objects
        to property |Element.model| creates some connection required by
        the respective model type automatically .  These
        examples  should be relevant for developers only.

        The following |hbranch| model branches a single input value
        (from to node `inp`) to multiple outputs (nodes `out1` and `out2`):

        >>> from hydpy import Element, Node, reverse_model_wildcard_import
        >>> reverse_model_wildcard_import()
        >>> element = Element('a_branch',
        ...                   inlets='branch_input',
        ...                   outlets=('branch_output_1', 'branch_output_2'))
        >>> inp = element.inlets.branch_input
        >>> out1, out2 = element.outlets
        >>> from hydpy.models.hbranch import *
        >>> parameterstep()
        >>> xpoints(0.0, 3.0)
        >>> ypoints(branch_output_1=[0.0, 1.0], branch_output_2=[0.0, 2.0])
        >>> parameters.update()
        >>> element.model = model

        To show that the inlet and outlet connections are built properly,
        we assign a new value to the inlet node `inp` and verify that the
        suitable fractions of this value are passed to the outlet nodes
        out1` and `out2` by calling method |Model.doit|:

        >>> inp.sequences.sim = 999.0
        >>> model.doit(0)
        >>> fluxes.input
        input(999.0)
        >>> out1.sequences.sim
        sim(333.0)
        >>> out2.sequences.sim
        sim(666.0)
        """
        model = vars(self).get('model')
        if model:
            return model
        raise AttributeError(
            f'The model object of element `{self.name}` has '
            f'been requested but not been prepared so far.')