def update(self):
        """Update value based on the actual |calc_qg_v1| method.

        Required derived parameter:
            |H|

        Note that the value of parameter |lstream_derived.QM| is directly
        related to the value of parameter |HM| and indirectly related to
        all parameters values relevant for method |calc_qg_v1|. Hence the
        complete paramter (and sequence) requirements might differ for
        various application models.

        For examples, see the documentation on method ToDo.
        """
        mod = self.subpars.pars.model
        con = mod.parameters.control
        flu = mod.sequences.fluxes
        flu.h = con.hm
        mod.calc_qg()
        self(flu.qg)