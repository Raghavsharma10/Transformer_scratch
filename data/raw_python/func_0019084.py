def calc_outuh_quh_v1(self):
    """Calculate the unit hydrograph output (convolution).

    Required derived parameters:
        |UH|

    Required flux sequences:
        |Q0|
        |Q1|
        |InUH|

    Updated log sequence:
        |QUH|

    Calculated flux sequence:
        |OutUH|

    Examples:

        Prepare a unit hydrograph with only three ordinates ---
        representing a fast catchment response compared to the selected
        step size:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> derived.uh.shape = 3
        >>> derived.uh = 0.3, 0.5, 0.2
        >>> logs.quh.shape = 3
        >>> logs.quh = 1.0, 3.0, 0.0

        Without new input, the actual output is simply the first value
        stored in the logging sequence and the values of the logging
        sequence are shifted to the left:

        >>> fluxes.inuh = 0.0
        >>> model.calc_outuh_quh_v1()
        >>> fluxes.outuh
        outuh(1.0)
        >>> logs.quh
        quh(3.0, 0.0, 0.0)

        With an new input of 4mm, the actual output consists of the first
        value stored in the logging sequence and the input value
        multiplied with the first unit hydrograph ordinate.  The updated
        logging sequence values result from the multiplication of the
        input values and the remaining ordinates:

        >>> fluxes.inuh = 4.0
        >>> model.calc_outuh_quh_v1()
        >>> fluxes.outuh
        outuh(4.2)
        >>> logs.quh
        quh(2.0, 0.8, 0.0)

        The next example demonstates the updating of non empty logging
        sequence:

        >>> fluxes.inuh = 4.0
        >>> model.calc_outuh_quh_v1()
        >>> fluxes.outuh
        outuh(3.2)
        >>> logs.quh
        quh(2.8, 0.8, 0.0)

        A unit hydrograph with only one ordinate results in the direct
        routing of the input:

        >>> derived.uh.shape = 1
        >>> derived.uh = 1.0
        >>> fluxes.inuh = 0.0
        >>> logs.quh.shape = 1
        >>> logs.quh = 0.0
        >>> model.calc_outuh_quh_v1()
        >>> fluxes.outuh
        outuh(0.0)
        >>> logs.quh
        quh(0.0)
        >>> fluxes.inuh = 4.0
        >>> model.calc_outuh_quh()
        >>> fluxes.outuh
        outuh(4.0)
        >>> logs.quh
        quh(0.0)
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    flu.outuh = der.uh[0]*flu.inuh+log.quh[0]
    for jdx in range(1, len(der.uh)):
        log.quh[jdx-1] = der.uh[jdx]*flu.inuh+log.quh[jdx]