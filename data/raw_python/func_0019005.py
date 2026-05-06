def calculate_single_terms(self):
        """Apply all methods stored in the hidden attribute
        `PART_ODE_METHODS`.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> k(0.25)
        >>> states.s = 1.0
        >>> model.calculate_single_terms()
        >>> fluxes.q
        q(0.25)
        """
        self.numvars.nmb_calls = self.numvars.nmb_calls+1
        for method in self.PART_ODE_METHODS:
            method(self)