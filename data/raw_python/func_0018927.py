def connect(self):
        """Connect the |LinkSequence| instances handled by the actual model
        to the |NodeSequence| instances handled by one inlet node and
        multiple oulet nodes.

        The HydPy-H-Branch model passes multiple output values to different
        outlet nodes.  This requires additional information regarding the
        `direction` of each output value.  Therefore, node names are used
        as keywords.  Assume the discharge values of both nodes `inflow1`
        and `inflow2`  shall be  branched to nodes `outflow1` and `outflow2`
        via element `branch`:

        >>> from hydpy import *
        >>> branch = Element('branch',
        ...                  inlets=['inflow1', 'inflow2'],
        ...                  outlets=['outflow1', 'outflow2'])

        Then parameter |YPoints| relates different supporting points via
        its keyword arguments to the respective nodes:

        >>> from hydpy.models.hbranch import *
        >>> parameterstep()
        >>> xpoints(0.0, 3.0)
        >>> ypoints(outflow1=[0.0, 1.0], outflow2=[0.0, 2.0])
        >>> parameters.update()

        After connecting the model with its element the total discharge
        value of nodes `inflow1` and `inflow2` can be properly divided:

        >>> branch.model = model
        >>> branch.inlets.inflow1.sequences.sim = 1.0
        >>> branch.inlets.inflow2.sequences.sim = 5.0
        >>> model.doit(0)
        >>> print(branch.outlets.outflow1.sequences.sim)
        sim(2.0)
        >>> print(branch.outlets.outflow2.sequences.sim)
        sim(4.0)

        In case of missing (or misspelled) outlet nodes, the following
        error is raised:

        >>> branch.outlets.mutable = True
        >>> del branch.outlets.outflow1
        >>> parameters.update()
        >>> model.connect()
        Traceback (most recent call last):
        ...
        RuntimeError: Model `hbranch` of element `branch` tried to connect \
to an outlet node named `outflow1`, which is not an available outlet node \
of element `branch`.
        """
        nodes = self.element.inlets
        total = self.sequences.inlets.total
        if total.shape != (len(nodes),):
            total.shape = len(nodes)
        for idx, node in enumerate(nodes):
            double = node.get_double('inlets')
            total.set_pointer(double, idx)
        for (idx, name) in enumerate(self.nodenames):
            try:
                outlet = getattr(self.element.outlets, name)
                double = outlet.get_double('outlets')
            except AttributeError:
                raise RuntimeError(
                    f'Model {objecttools.elementphrase(self)} tried '
                    f'to connect to an outlet node named `{name}`, '
                    f'which is not an available outlet node of element '
                    f'`{self.element.name}`.')
            self.sequences.outlets.branched.set_pointer(double, idx)