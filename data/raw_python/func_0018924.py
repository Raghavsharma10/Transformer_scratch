def calc_outputs_v1(self):
    """Performs the actual interpolation or extrapolation.

    Required control parameters:
      |XPoints|
      |YPoints|

    Required derived parameter:
      |NmbPoints|
      |NmbBranches|

    Required flux sequence:
      |Input|

    Calculated flux sequence:
      |Outputs|

    Examples:

        As a simple example, assume a weir directing all discharge into
        `branch1` until the capacity limit of 2 m³/s is reached.  The
        discharge exceeding this threshold is directed into `branch2`:

        >>> from hydpy.models.hbranch import *
        >>> parameterstep()
        >>> xpoints(0., 2., 4.)
        >>> ypoints(branch1=[0., 2., 2.],
        ...         branch2=[0., 0., 2.])
        >>> model.parameters.update()

        Low discharge example (linear interpolation between the first two
        supporting point pairs):

        >>> fluxes.input = 1.
        >>> model.calc_outputs_v1()
        >>> fluxes.outputs
        outputs(branch1=1.0,
                branch2=0.0)

        Medium discharge example (linear interpolation between the second
        two supporting point pairs):

        >>> fluxes.input = 3.
        >>> model.calc_outputs_v1()
        >>> print(fluxes.outputs)
        outputs(branch1=2.0,
                branch2=1.0)

        High discharge example (linear extrapolation beyond the second two
        supporting point pairs):

        >>> fluxes.input = 5.
        >>> model.calc_outputs_v1()
        >>> fluxes.outputs
        outputs(branch1=2.0,
                branch2=3.0)

        Non-monotonous relationships and balance violations are allowed,
        e.g.:

        >>> xpoints(0., 2., 4., 6.)
        >>> ypoints(branch1=[0., 2., 0., 0.],
        ...         branch2=[0., 0., 2., 4.])
        >>> model.parameters.update()
        >>> fluxes.input = 7.
        >>> model.calc_outputs_v1()
        >>> fluxes.outputs
        outputs(branch1=0.0,
                branch2=5.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    # Search for the index of the two relevant x points...
    for pdx in range(1, der.nmbpoints):
        if con.xpoints[pdx] > flu.input:
            break
    # ...and use it for linear interpolation (or extrapolation).
    for bdx in range(der.nmbbranches):
        flu.outputs[bdx] = (
            (flu.input-con.xpoints[pdx-1]) *
            (con.ypoints[bdx, pdx]-con.ypoints[bdx, pdx-1]) /
            (con.xpoints[pdx]-con.xpoints[pdx-1]) +
            con.ypoints[bdx, pdx-1])