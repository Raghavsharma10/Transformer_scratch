def softcal(self, std, dest, serial, T1, U1, T2, U2, T3=None, U3=None):
        """Generates a softcal curve.

        :param std: The standard curve index used to calculate the softcal
            curve. Valid entries are 1-20
        :param dest: The user curve index where the softcal curve is stored.
            Valid entries are 21-60.
        :param serial: The serial number of the new curve. A maximum of 10
            characters is allowed.
        :param T1: The first temperature point.
        :param U1: The first sensor units point.
        :param T2: The second temperature point.
        :param U2: The second sensor units point.
        :param T3: The third temperature point. Default: `None`.
        :param U3: The third sensor units point. Default: `None`.

        """
        args = [std, dest, serial, T1, U1, T2, U2]
        dtype = [Integer(min=1, max=21), Integer(min=21, max=61), String(max=10), Float, Float, Float, Float]
        if (T3 is not None) and (U3 is not None):
            args.extend([T3, U3])
            dtype.extend([Float, Float])
        self._write(('SCAL', dtype), *args)