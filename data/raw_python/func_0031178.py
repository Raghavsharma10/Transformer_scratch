def registerSimulator(self, name=None, hdl=None, analyze_cmd=None, elaborate_cmd=None, simulate_cmd=None):
        ''' Registers an HDL _simulator
                name - str, user defined name, used to identify this _simulator record
                hdl - str, case insensitive, (verilog, vhdl), the HDL to which the simulated MyHDL code will be converted
                analyze_cmd - str, system command that will be run to analyze the generated HDL
                elaborate_cmd - str, optional, system command that will be run after the analyze phase
                simulate_cmd - str, system command that will be run to simulate the analyzed and elaborated design
                Before execution of a command string the following substitutions take place:
                    {topname} is substituted with the name of the simulated MyHDL function
        '''
        if not isinstance(name, str) or (name.strip() == ""):
            raise ValueError("Invalid _simulator name")
        if hdl.lower() not in ("vhdl", "verilog"):
            raise ValueError("Invalid hdl {}".format(hdl))
        if not isinstance(analyze_cmd, str) or (analyze_cmd.strip() == ""):
            raise ValueError("Invalid analyzer command")
        if elaborate_cmd is not None:
            if not isinstance(elaborate_cmd, str) or (elaborate_cmd.strip() == ""):
                raise ValueError("Invalid elaborate_cmd command")
        if not isinstance(simulate_cmd, str) or (simulate_cmd.strip() == ""):
            raise ValueError("Invalid _simulator command")

        self.sim_reg[name] = (hdl.lower(), analyze_cmd, elaborate_cmd, simulate_cmd)