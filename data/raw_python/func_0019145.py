def update(self):
        """Update |UH| based on |MaxBaz|.

        .. note::

            This method also updates the shape of log sequence |QUH|.

        |MaxBaz| determines the end point of the triangle.  A value of
        |MaxBaz| being not larger than the simulation step size is
        identical with applying no unit hydrograph at all:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> maxbaz(0.0)
        >>> derived.uh.update()
        >>> logs.quh.shape
        (1,)
        >>> derived.uh
        uh(1.0)

        Note that, due to difference of the parameter and the simulation
        step size in the given example, the largest assignment resulting
        in a `inactive` unit hydrograph is 1/2:

        >>> maxbaz(0.5)
        >>> derived.uh.update()
        >>> logs.quh.shape
        (1,)
        >>> derived.uh
        uh(1.0)

        When |MaxBaz| is in accordance with two simulation steps, both
        unit hydrograph ordinats must be 1/2 due to symmetry of the
        triangle:

        >>> maxbaz(1.0)
        >>> derived.uh.update()
        >>> logs.quh.shape
        (2,)
        >>> derived.uh
        uh(0.5)
        >>> derived.uh.values
        array([ 0.5,  0.5])

        A |MaxBaz| value in accordance with three simulation steps results
        in the ordinate values 2/9, 5/9, and 2/9:

        >>> maxbaz(1.5)
        >>> derived.uh.update()
        >>> logs.quh.shape
        (3,)
        >>> derived.uh
        uh(0.222222, 0.555556, 0.222222)

        And a final example, where the end of the triangle lies within
        a simulation step, resulting in the fractions 8/49, 23/49, 16/49,
        and 2/49:

        >>> maxbaz(1.75)
        >>> derived.uh.update()
        >>> logs.quh.shape
        (4,)
        >>> derived.uh
        uh(0.163265, 0.469388, 0.326531, 0.040816)
        """
        maxbaz = self.subpars.pars.control.maxbaz.value
        quh = self.subpars.pars.model.sequences.logs.quh
        # Determine UH parameters...
        if maxbaz <= 1.:
            # ...when MaxBaz smaller than or equal to the simulation time step.
            self.shape = 1
            self(1.)
            quh.shape = 1
        else:
            # ...when MaxBaz is greater than the simulation time step.
            # Define some shortcuts for the following calculations.
            full = maxbaz
            # Now comes a terrible trick due to rounding problems coming from
            # the conversation of the SMHI parameter set to the HydPy
            # parameter set.  Time to get rid of it...
            if (full % 1.) < 1e-4:
                full //= 1.
            full_f = int(numpy.floor(full))
            full_c = int(numpy.ceil(full))
            half = full/2.
            half_f = int(numpy.floor(half))
            half_c = int(numpy.ceil(half))
            full_2 = full**2.
            # Calculate the triangle ordinate(s)...
            self.shape = full_c
            uh = self.values
            quh.shape = full_c
            # ...of the rising limb.
            points = numpy.arange(1, half_f+1)
            uh[:half_f] = (2.*points-1.)/(2.*full_2)
            # ...around the peak (if it exists).
            if numpy.mod(half, 1.) != 0.:
                uh[half_f] = (
                    (half_c-half)/full +
                    (2*half**2.-half_f**2.-half_c**2.)/(2.*full_2))
            # ...of the falling limb (eventually except the last one).
            points = numpy.arange(half_c+1., full_f+1.)
            uh[half_c:full_f] = 1./full-(2.*points-1.)/(2.*full_2)
            # ...at the end (if not already done).
            if numpy.mod(full, 1.) != 0.:
                uh[full_f] = (
                    (full-full_f)/full-(full_2-full_f**2.)/(2.*full_2))
            # Normalize the ordinates.
            self(uh/numpy.sum(uh))