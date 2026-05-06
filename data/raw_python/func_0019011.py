def extrapolate_error(self):
        """Estimate the numerical error to be expected when applying all
        methods available based on the results of the current and the
        last method.

        Note that this expolation strategy cannot be applied on the first
        method.  If the current method is the first one, `-999.9` is returned.

        >>> from hydpy.models.test_v1 import *
        >>> parameterstep()
        >>> model.numvars.error = 1e-2
        >>> model.numvars.last_error = 1e-1
        >>> model.numvars.idx_method = 10
        >>> model.extrapolate_error()
        >>> from hydpy import round_
        >>> round_(model.numvars.extrapolated_error)
        0.01
        >>> model.numvars.idx_method = 9
        >>> model.extrapolate_error()
        >>> round_(model.numvars.extrapolated_error)
        0.001
        """
        if self.numvars.idx_method > 2:
            self.numvars.extrapolated_error = modelutils.exp(
                modelutils.log(self.numvars.error) +
                (modelutils.log(self.numvars.error) -
                 modelutils.log(self.numvars.last_error)) *
                (self.numconsts.nmb_methods-self.numvars.idx_method))
        else:
            self.numvars.extrapolated_error = -999.9