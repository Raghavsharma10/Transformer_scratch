def calc_flooddischarge_v1(self):
    """Calculate the discharge during and after a flood event based on an
    |anntools.SeasonalANN| describing the relationship(s) between discharge
    and water stage.

    Required control parameter:
      |WaterLevel2FloodDischarge|

    Required derived parameter:
      |dam_derived.TOY|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |FloodDischarge|

    Example:


        The control parameter |WaterLevel2FloodDischarge| is derived from
        |SeasonalParameter|.  This allows to simulate different seasonal
        dam control schemes.  To show that the seasonal selection mechanism
        is implemented properly, we define a short simulation period of
        three days:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.01.01', '2001.01.04', '1d'

        Now we prepare a dam model and define two different relationships
        between water level and flood discharge.  The first relatively
        simple relationship (for January, 2) is based on two neurons
        contained in a single hidden layer and is used in the following
        example.  The second neural network (for January, 3) is not
        applied at all, which is why we do not need to assign any parameter
        values to it:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> waterlevel2flooddischarge(
        ...     _01_02_12 = ann(nmb_inputs=1,
        ...                     nmb_neurons=(2,),
        ...                     nmb_outputs=1,
        ...                     weights_input=[[50., 4]],
        ...                     weights_output=[[2.], [30]],
        ...                     intercepts_hidden=[[-13000, -1046]],
        ...                     intercepts_output=[0.]),
        ...     _01_03_12 = ann(nmb_inputs=1,
        ...                     nmb_neurons=(2,),
        ...                     nmb_outputs=1))
        >>> derived.toy.update()
        >>> model.idx_sim = pub.timegrids.sim['2001.01.02']

        The following example shows two distinct effects of both neurons
        in the first network.  One neuron describes a relatively sharp
        increase between 259.8 and 260.2 meters from about 0 to 2 m³/s.
        This could describe a release of water through a bottom outlet
        controlled by a valve.  The add something like an exponential
        increase between 260 and 261 meters, which could describe the
        uncontrolled flow over a spillway:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_flooddischarge_v1,
        ...                 last_example=21,
        ...                 parseqs=(aides.waterlevel,
        ...                          fluxes.flooddischarge))
        >>> test.nexts.waterlevel = numpy.arange(257, 261.1, 0.2)
        >>> test()
        | ex. | waterlevel | flooddischarge |
        -------------------------------------
        |   1 |      257.0 |            0.0 |
        |   2 |      257.2 |       0.000001 |
        |   3 |      257.4 |       0.000002 |
        |   4 |      257.6 |       0.000005 |
        |   5 |      257.8 |       0.000011 |
        |   6 |      258.0 |       0.000025 |
        |   7 |      258.2 |       0.000056 |
        |   8 |      258.4 |       0.000124 |
        |   9 |      258.6 |       0.000275 |
        |  10 |      258.8 |       0.000612 |
        |  11 |      259.0 |       0.001362 |
        |  12 |      259.2 |       0.003031 |
        |  13 |      259.4 |       0.006745 |
        |  14 |      259.6 |       0.015006 |
        |  15 |      259.8 |       0.033467 |
        |  16 |      260.0 |       1.074179 |
        |  17 |      260.2 |       2.164498 |
        |  18 |      260.4 |       2.363853 |
        |  19 |      260.6 |        2.79791 |
        |  20 |      260.8 |       3.719725 |
        |  21 |      261.0 |       5.576088 |
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    con.waterlevel2flooddischarge.inputs[0] = aid.waterlevel
    con.waterlevel2flooddischarge.process_actual_input(der.toy[self.idx_sim])
    flu.flooddischarge = con.waterlevel2flooddischarge.outputs[0]