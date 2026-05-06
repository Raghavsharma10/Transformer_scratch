def simindices(self):
        """Tuple containing the start and end index of the simulation period
        regarding the initialization period defined by the |Timegrids| object
        stored in module |pub|."""
        return (hydpy.pub.timegrids.init[hydpy.pub.timegrids.sim.firstdate],
                hydpy.pub.timegrids.init[hydpy.pub.timegrids.sim.lastdate])