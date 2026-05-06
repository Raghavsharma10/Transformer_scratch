def fourier_ratios(phase_shifted_coeffs):
        r"""
        Returns the :math:`R_{j1}` and :math:`\phi_{j1}` values for the given
        phase-shifted coefficients.

        .. math::

            R_{j1} = A_j / A_1

        .. math::

            \phi_{j1} = \phi_j - j \phi_1

        **Parameters**

        phase_shifted_coeffs : array-like, shape = [:math:`2n+1`]
            Fourier sine or cosine series coefficients.
            :math:`[ A_0, A_1, \Phi_1, \ldots, A_n, \Phi_n ]`.

        **Returns**

        out : array-like, shape = [:math:`2n+1`]
            Fourier ratios
            :math:`[ R_{21}, \phi_{21}, \ldots, R_{n1}, \phi_{n1} ]`.
        """


        n_coeff = phase_shifted_coeffs.size
        # n_coeff = 2*degree + 1 => degree = (n_coeff-1)/2
        degree = (n_coeff - 1) / 2

        amplitudes = phase_shifted_coeffs[1::2]
        phases = phase_shifted_coeffs[2::2]

        # there are degree-1 amplitude ratios, and degree-1 phase deltas,
        # so altogether there are 2*(degree-1) values
        ratios = numpy.empty(2*(degree-1), dtype=float)
        amplitude_ratios = ratios[::2]
        phase_deltas = ratios[1::2]

        # amplitudes may be zero, so suppress division by zero warnings
        with numpy.errstate(divide="ignore"):
            amplitude_ratios[:] = amplitudes[1:]
            amplitude_ratios   /= amplitudes[0]

        # indices for phase deltas
        i = numpy.arange(2, degree+1)
        phase_deltas[:] = phases[1:]
        phase_deltas   -= i*phases[0]
        # constrain phase_deltas between 0 and 2*pi
        phase_deltas   %= 2*pi

        return ratios