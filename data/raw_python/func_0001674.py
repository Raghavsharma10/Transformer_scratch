def init_runner(self, parser, tracers, projinfo):
        ''' initial some instances for preparing to run test case
        @note:  should not override
        @param parser: instance of TestCaseParser
        @param tracers: dict type for the instance of Tracer. Such as {"":tracer_obj} or {"192.168.0.1:5555":tracer_obj1, "192.168.0.2:5555":tracer_obj2} 
        @param proj_info: dict type of test case.  use like:  self.proj_info["module"], self.proj_info["name"]
            yaml case like: 
                - project:
                    name: xxx
                    module: xxxx
            dict case like:
                {"project": {"name": xxx, "module": xxxx}}            
                
        '''
        self.parser = parser
        self.tracers = tracers
        self.proj_info = projinfo