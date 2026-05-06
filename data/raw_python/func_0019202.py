def calc_possibleremoterelieve_v1(self):
    """Calculate the highest possible water release that can be routed to
    a remote location based on an artificial neural network describing the
    relationship between possible release and water stage.

    Required control parameter:
      |WaterLevel2PossibleRemoteRelieve|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |PossibleRemoteRelieve|

    Example:

        For simplicity, the example of method |calc_flooddischarge_v1|
        is reused.  See the documentation on the mentioned method for
        further information:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> waterlevel2possibleremoterelieve(
        ...     nmb_inputs=1,
        ...     nmb_neurons=(2,),
        ...     nmb_outputs=1,
        ...     weights_input=[[50., 4]],
        ...     weights_output=[[2.], [30]],
        ...     intercepts_hidden=[[-13000, -1046]],
        ...     intercepts_output=[0.])
        >>> from hydpy import UnitTest
        >>> test = UnitTest(
        ...     model, model.calc_possibleremoterelieve_v1,
        ...     last_example=21,
        ...     parseqs=(aides.waterlevel, fluxes.possibleremoterelieve))
        >>> test.nexts.waterlevel = numpy.arange(257, 261.1, 0.2)
        >>> test()
        | ex. | waterlevel | possibleremoterelieve |
        --------------------------------------------
        |   1 |      257.0 |                   0.0 |
        |   2 |      257.2 |              0.000001 |
        |   3 |      257.4 |              0.000002 |
        |   4 |      257.6 |              0.000005 |
        |   5 |      257.8 |              0.000011 |
        |   6 |      258.0 |              0.000025 |
        |   7 |      258.2 |              0.000056 |
        |   8 |      258.4 |              0.000124 |
        |   9 |      258.6 |              0.000275 |
        |  10 |      258.8 |              0.000612 |
        |  11 |      259.0 |              0.001362 |
        |  12 |      259.2 |              0.003031 |
        |  13 |      259.4 |              0.006745 |
        |  14 |      259.6 |              0.015006 |
        |  15 |      259.8 |              0.033467 |
        |  16 |      260.0 |              1.074179 |
        |  17 |      260.2 |              2.164498 |
        |  18 |      260.4 |              2.363853 |
        |  19 |      260.6 |               2.79791 |
        |  20 |      260.8 |              3.719725 |
        |  21 |      261.0 |              5.576088 |
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    con.waterlevel2possibleremoterelieve.inputs[0] = aid.waterlevel
    con.waterlevel2possibleremoterelieve.process_actual_input()
    flu.possibleremoterelieve = con.waterlevel2possibleremoterelieve.outputs[0]