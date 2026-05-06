def update(self):
        """Determines in how many segments the whole reach needs to be
        divided to approximate the desired lag time via integer rounding.
        Adjusts the shape of sequence |QJoints| additionally.

        Required control parameters:
          |Lag|

        Calculated derived parameters:
          |NmbSegments|

        Prepared state sequence:
          |QJoints|

        Examples:

            Define a lag time of 1.4 days and a simulation step size of 12
            hours:

            >>> from hydpy.models.hstream import *
            >>> parameterstep('1d')
            >>> simulationstep('12h')
            >>> lag(1.4)

            Then the actual lag value for the simulation step size is 2.8

            >>> lag
            lag(1.4)
            >>> lag.value
            2.8

            Through rounding the number of segments is determined:

            >>> derived.nmbsegments.update()
            >>> derived.nmbsegments
            nmbsegments(3)

            The number of joints is always the number of segments plus one:

            >>> states.qjoints.shape
            (4,)
        """
        pars = self.subpars.pars
        self(int(round(pars.control.lag)))
        pars.model.sequences.states.qjoints.shape = self+1