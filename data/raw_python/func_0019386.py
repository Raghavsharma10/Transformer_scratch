def update(self):
        """Determine the number of substeps.

        Initialize a llake model and assume a simulation step size of 12 hours:

        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')

        If the maximum internal step size is also set to 12 hours, there is
        only one internal calculation step per outer simulation step:

        >>> maxdt('12h')
        >>> derived.nmbsubsteps.update()
        >>> derived.nmbsubsteps
        nmbsubsteps(1)

        Assigning smaller values to `maxdt` increases `nmbstepsize`:

        >>> maxdt('1h')
        >>> derived.nmbsubsteps.update()
        >>> derived.nmbsubsteps
        nmbsubsteps(12)

        In case the simulationstep is not a whole multiple of `dwmax`,
        the value of `nmbsubsteps` is rounded up:

        >>> maxdt('59m')
        >>> derived.nmbsubsteps.update()
        >>> derived.nmbsubsteps
        nmbsubsteps(13)

        Even for `maxdt` values exceeding the simulationstep, the value
        of `numbsubsteps` does not become smaller than one:

        >>> maxdt('2d')
        >>> derived.nmbsubsteps.update()
        >>> derived.nmbsubsteps
        nmbsubsteps(1)
        """
        maxdt = self.subpars.pars.control.maxdt
        seconds = self.simulationstep.seconds
        self.value = numpy.ceil(seconds/maxdt)