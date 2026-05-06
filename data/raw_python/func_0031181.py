def _getDut(self, func, **kwargs):
        ''' Returns a simulation instance of func. 
            Uses the simulator specified by self._simulator. 
            Enables traces if self._trace is True
                func - MyHDL function to be simulated
                kwargs - dict of func interface assignments: for signals and parameters
        '''
        if self._simulator=="myhdl":
            if not self._trace:
                sim_dut = func(**kwargs)
            else:
                sim_dut = traceSignals(func, **kwargs)
        else:
            sim_dut = self._getCosimulation(func, **kwargs)

        return sim_dut