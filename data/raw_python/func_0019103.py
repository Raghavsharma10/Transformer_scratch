def calc_v_qa_v1(self):
    """Update the stored water volume based on the equation of continuity.

    Note that for too high outflow values, which would result in overdraining
    the lake, the outflow is trimmed.

    Required derived parameters:
      |Seconds|
      |NmbSubsteps|

    Required flux sequence:
      |QZ|

    Updated aide sequences:
      |llake_aides.QA|
      |llake_aides.V|

    Basic Equation:
      :math:`\\frac{dV}{dt}= QZ - QA`

    Examples:

        Prepare a lake model with an initial storage of 100.000 m³ and an
        inflow of 2 m³/s and a (potential) outflow of 6 m³/s:

        >>> from hydpy.models.llake import *
        >>> parameterstep()
        >>> simulationstep('12h')
        >>> maxdt('6h')
        >>> derived.seconds.update()
        >>> derived.nmbsubsteps.update()
        >>> aides.v = 1e5
        >>> fluxes.qz = 2.
        >>> aides.qa = 6.

        Through calling method `calc_v_qa_v1` three times with the same inflow
        and outflow values, the storage is emptied after the second step and
        outflow is equal to inflow after the third step:

        >>> model.calc_v_qa_v1()
        >>> aides.v
        v(13600.0)
        >>> aides.qa
        qa(6.0)
        >>> model.new2old()
        >>> model.calc_v_qa_v1()
        >>> aides.v
        v(0.0)
        >>> aides.qa
        qa(2.62963)
        >>> model.new2old()
        >>> model.calc_v_qa_v1()
        >>> aides.v
        v(0.0)
        >>> aides.qa
        qa(2.0)

        Note that the results of method |calc_v_qa_v1| are not based
        depend on the (outer) simulation step size but on the (inner)
        calculation step size defined by parameter `maxdt`.
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    aid.qa = min(aid.qa, flu.qz+der.nmbsubsteps/der.seconds*aid.v)
    aid.v = max(aid.v+der.seconds/der.nmbsubsteps*(flu.qz-aid.qa), 0.)