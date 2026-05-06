def _getCosimulation(self, func, **kwargs):
        ''' Returns a co-simulation instance of func. 
            Uses the _simulator specified by self._simulator. 
            Enables traces if self._trace is True
                func - MyHDL function to be simulated
                kwargs - dict of func interface assignments: for signals and parameters
        '''
        vals = {}
        vals['topname'] = func.func_name
        vals['unitname'] = func.func_name.lower()
        hdlsim = self._simulator
        if not hdlsim:
            raise ValueError("No _simulator specified")
        if not self.sim_reg.has_key(hdlsim):
            raise ValueError("Simulator {} is not registered".format(hdlsim))

        hdl, analyze_cmd, elaborate_cmd, simulate_cmd = self.sim_reg[hdlsim]

        # Convert to HDL
        if hdl == "verilog":
            toVerilog(func, **kwargs)
            if self._trace:
                self._enableTracesVerilog("./tb_{topname}.v".format(**vals))
        elif hdl == "vhdl":
            toVHDL(func, **kwargs)

        # Analyze HDL
        os.system(analyze_cmd.format(**vals))
        # Elaborate
        if elaborate_cmd:
            os.system(elaborate_cmd.format(**vals))
        # Simulate
        return Cosimulation(simulate_cmd.format(**vals), **kwargs)