def ebeam(self, context):
        """ ebeam cube data source """
        if context.shape != self.shape:
            raise ValueError("Partial feeding of the "
                "beam cube is not yet supported %s %s." % (context.shape, self.shape))

        ebeam = np.empty(context.shape, context.dtype)

        # Iterate through the correlations,
        # assigning real and imaginary data, if present,
        # otherwise zeroing the correlation
        for i, (re, im) in enumerate(self._files.itervalues()):
            ebeam[:,:,:,i].real[:] = 0 if re is None else re[0].data.T
            ebeam[:,:,:,i].imag[:] = 0 if im is None else im[0].data.T

        return ebeam