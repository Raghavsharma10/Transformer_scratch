def phase_shifted_coefficients(amplitude_coefficients, form='cos',
                                   shift=0.0):
        r"""
        Converts Fourier coefficients from the amplitude form to the
        phase-shifted form, as either a sine or cosine series.

        Amplitude form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n (a_k \sin(k \omega t)
                                     + b_k \cos(k \omega t))

        Sine form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n A_k \sin(k \omega t + \Phi_k)

        Cosine form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n A_k \cos(k \omega t + \Phi_k)

        **Parameters**

        amplitude_coefficients : array-like, shape = [:math:`2n+1`]
            Array of coefficients
            :math:`[ A_0, a_1, b_1, \ldots a_n, b_n ]`.
        form : str, optional
            Form of output coefficients, must be one of 'sin' or 'cos'
            (default 'cos').
        shift : number, optional
            Shift to apply to light curve (default 0.0).

        **Returns**

        out : array-like, shape = [:math:`2n+1`]
            Array of coefficients
            :math:`[ A_0, A_1, \Phi_1, \ldots, A_n, \Phi_n ]`.
        """
        if form != 'sin' and form != 'cos':
            raise NotImplementedError(
                'Fourier series must have form sin or cos')

        # separate array of coefficients into respective parts
        A_0 = amplitude_coefficients[0]
        a_k = amplitude_coefficients[1::2]
        b_k = amplitude_coefficients[2::2]

        degree = a_k.size
        k = numpy.arange(1, degree+1)
        # A_k and Phi_k are the angle and hypotenuse in the right triangles
        # pictured below. A_k is obtained with the Pythagorean theorem, and
        # Phi_k is obtained with the 2-argument inverse tangent.
        # The positions of a_k and b_k depend on whether it is a sin or cos
        # series.
        #
        # Cos series                Sin series
        #
        #    b_k                          /|
        # ---------                      / |
        # \ Φ_k |_|                     /  |
        #  \      |                A_k /   |
        #   \     |                   /    | b_k
        #    \    | a_k              /     |
        # A_k \   |                 /     _|
        #      \  |                / Φ_k | |
        #       \ |                ---------
        #        \|                   a_k
        #
        A_k   = numpy.sqrt(a_k**2 + b_k**2)
        # phase coefficients are shifted to the left by optional ``shift``
        if form == 'cos':
            Phi_k = numpy.arctan2(-a_k, b_k) + 2*pi*k*shift
        elif form == 'sin':
            Phi_k = numpy.arctan2(b_k, a_k) + 2*pi*k*shift
        # constrain Phi between 0 and 2*pi
        Phi_k %= 2*pi

        phase_shifted_coefficients_ = numpy.empty(amplitude_coefficients.shape,
                                                  dtype=float)
        phase_shifted_coefficients_[0]    = A_0
        phase_shifted_coefficients_[1::2] = A_k
        phase_shifted_coefficients_[2::2] = Phi_k

        return phase_shifted_coefficients_