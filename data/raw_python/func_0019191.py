def calc_waterlevel_v1(self):
    """Determine the water level based on an artificial neural network
    describing the relationship between water level and water stage.

    Required control parameter:
      |WaterVolume2WaterLevel|

    Required state sequence:
      |WaterVolume|

    Calculated aide sequence:
      |WaterLevel|

    Example:

        Prepare a dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()

        Prepare a very simple relationship based on one single neuron:

        >>> watervolume2waterlevel(
        ...         nmb_inputs=1, nmb_neurons=(1,), nmb_outputs=1,
        ...         weights_input=0.5, weights_output=1.0,
        ...         intercepts_hidden=0.0, intercepts_output=-0.5)

        At least in the water volume range used in the following examples,
        the shape of the relationship looks acceptable:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(
        ...     model, model.calc_waterlevel_v1,
        ...     last_example=10,
        ...     parseqs=(states.watervolume, aides.waterlevel))
        >>> test.nexts.watervolume = range(10)
        >>> test()
        | ex. | watervolume | waterlevel |
        ----------------------------------
        |   1 |         0.0 |        0.0 |
        |   2 |         1.0 |   0.122459 |
        |   3 |         2.0 |   0.231059 |
        |   4 |         3.0 |   0.317574 |
        |   5 |         4.0 |   0.380797 |
        |   6 |         5.0 |   0.424142 |
        |   7 |         6.0 |   0.452574 |
        |   8 |         7.0 |   0.470688 |
        |   9 |         8.0 |   0.482014 |
        |  10 |         9.0 |   0.489013 |

        For more realistic approximations of measured relationships between
        water level and volume, larger neural networks are required.
    """
    con = self.parameters.control.fastaccess
    new = self.sequences.states.fastaccess_new
    aid = self.sequences.aides.fastaccess
    con.watervolume2waterlevel.inputs[0] = new.watervolume
    con.watervolume2waterlevel.process_actual_input()
    aid.waterlevel = con.watervolume2waterlevel.outputs[0]