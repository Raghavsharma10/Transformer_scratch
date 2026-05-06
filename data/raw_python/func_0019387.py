def update(self):
        """Calulate the auxilary term.

        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> n(3)
        >>> v(0., 1e5, 1e6)
        >>> q(_1=[0., 1., 2.], _7=[0., 2., 5.])
        >>> maxdt('12h')
        >>> derived.seconds.update()
        >>> derived.nmbsubsteps.update()
        >>> derived.vq.update()
        >>> derived.vq
        vq(toy_1_1_0_0_0=[0.0, 243200.0, 2086400.0],
           toy_7_1_0_0_0=[0.0, 286400.0, 2216000.0])
        """
        con = self.subpars.pars.control
        der = self.subpars
        for (toy, qs) in con.q:
            setattr(self, str(toy), 2.*con.v+der.seconds/der.nmbsubsteps*qs)
        self.refresh()