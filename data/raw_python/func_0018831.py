def POST_timegrid(self) -> None:
        """Change the current simulation |Timegrid|."""
        init = hydpy.pub.timegrids.init
        sim = hydpy.pub.timegrids.sim
        sim.firstdate = self._inputs['firstdate']
        sim.lastdate = self._inputs['lastdate']
        state.idx1 = init[sim.firstdate]
        state.idx2 = init[sim.lastdate]