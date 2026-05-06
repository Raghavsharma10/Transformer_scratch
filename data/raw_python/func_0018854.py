def update(self) -> None:
        """Call method |Parameter.update| of all "secondary" parameters.

        Directly after initialisation, neither the primary (`control`)
        parameters nor the secondary (`derived`)  parameters of
        application model |hstream_v1| are ready for usage:

        >>> from hydpy.models.hstream_v1 import *
        >>> parameterstep('1d')
        >>> simulationstep('1d')
        >>> derived
        nmbsegments(?)
        c1(?)
        c3(?)
        c2(?)

        Trying to update the values of the secondary parameters while the
        primary ones are still not defined, raises errors like the following:

        >>> model.parameters.update()
        Traceback (most recent call last):
        ...
        AttributeError: While trying to update parameter ``nmbsegments` \
of element `?``, the following error occurred: For variable `lag`, \
no value has been defined so far.

        With proper values both for parameter |hstream_control.Lag| and
        |hstream_control.Damp|, updating the derived parameters succeeds:

        >>> lag(0.0)
        >>> damp(0.0)
        >>> model.parameters.update()
        >>> derived
        nmbsegments(0)
        c1(0.0)
        c3(0.0)
        c2(1.0)
        """
        for subpars in self.secondary_subpars:
            for par in subpars:
                try:
                    par.update()
                except BaseException:
                    objecttools.augment_excmessage(
                        f'While trying to update parameter '
                        f'`{objecttools.elementphrase(par)}`')