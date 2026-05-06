def design_matrix(phases, degree):
        r"""
        Constructs an :math:`N \times 2n+1` matrix of the form:

        .. math::

            \begin{bmatrix}
              1
            & \sin(1 \cdot 2\pi \cdot \phi_0)
            & \cos(1 \cdot 2\pi \cdot \phi_0)
            & \ldots
            & \sin(n \cdot 2\pi \cdot \phi_0)
            & \cos(n \cdot 2\pi \cdot \phi_0)
            \\
              \vdots
            & \vdots
            & \vdots
            & \ddots
            & \vdots
            & \vdots
            \\
              1
            & \sin(1 \cdot 2\pi \cdot \phi_N)
            & \cos(1 \cdot 2\pi \cdot \phi_N)
            & \ldots
            & \sin(n \cdot 2\pi \cdot \phi_N)
            & \cos(n \cdot 2\pi \cdot \phi_N)
            \end{bmatrix}

        where :math:`n =` *degree*, :math:`N =` *n_samples*, and
        :math:`\phi_i =` *phases[i]*.

        Parameters
        ----------
        phases : array-like, shape = [n_samples]
            
        """
        n_samples = phases.size
        # initialize coefficient matrix
        M = numpy.empty((n_samples, 2*degree+1))
        # indices
        i = numpy.arange(1, degree+1)
        # initialize the Nxn matrix that is repeated within the
        # sine and cosine terms
        x = numpy.empty((n_samples, degree))
        # the Nxn matrix now has N copies of the same row, and each row is
        # integer multiples of pi counting from 1 to the degree
        x[:,:] = i*2*numpy.pi
        # multiply each row of x by the phases
        x.T[:,:] *= phases
        # place 1's in the first column of the coefficient matrix
        M[:,0]    = 1
        # the odd indices of the coefficient matrix have sine terms
        M[:,1::2] = numpy.sin(x)
        # the even indices of the coefficient matrix have cosine terms
        M[:,2::2] = numpy.cos(x)
        return M