def modify_qa_v1(self):
    """Add water to or remove water from the calculated lake outflow.

    Required control parameter:
      |Verzw|

    Required derived parameter:
      |llake_derived.TOY|

    Updated flux sequence:
      |llake_fluxes.QA|

    Basic Equation:
      :math:`QA = QA* - Verzw`

    Examples:
        In preparation for the following examples, define a short simulation
        time period with a simulation step size of 12 hours and initialize
        the required model object:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000.01.01', '2000.01.04', '12h'
        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> derived.toy.update()

        Select the first half of the second day of January as the simulation
        step relevant for the following examples:

        >>> model.idx_sim = pub.timegrids.init['2000.01.02']

        Assume that, in accordance with previous calculations, the original
        outflow value is 3 m³/s:

        >>> fluxes.qa = 3.

        Prepare the shape of parameter `verzw` (usually, this is done
        automatically when calling parameter `n`):
        >>> verzw.shape = (None,)

        Set the value of the abstraction on the first half of the second
        day of January to 2 m³/s:

        >>> verzw(_1_1_18=0.,
        ...       _1_2_6=2.,
        ...       _1_2_18=0.)

        In the first example `verzw` is simply subtracted from `qa`:

        >>> model.modify_qa_v1()
        >>> fluxes.qa
        qa(1.0)

        In the second example `verzw` exceeds `qa`, resulting in a zero
        outflow value:

        >>> model.modify_qa_v1()
        >>> fluxes.qa
        qa(0.0)

        The last example demonstrates, that "negative abstractions" are
        allowed, resulting in an increase in simulated outflow:

        >>> verzw.toy_1_2_6 = -2.
        >>> model.modify_qa_v1()
        >>> fluxes.qa
        qa(2.0)
    """

    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    idx = der.toy[self.idx_sim]
    flu.qa = max(flu.qa-con.verzw[idx], 0.)