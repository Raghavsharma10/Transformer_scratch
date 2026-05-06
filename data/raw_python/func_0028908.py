def STM(self, params):
        """
        STM Ra!, {RLoList}

        Store multiple registers into memory
        """
        # TODO what registers can be stored?
        Ra, RLoList = self.get_two_parameters(r'\s*([^\s,]*)!,\s*{(.*)}(.*)', params).split(',')
        RLoList = RLoList.split(',')
        RLoList = [i.strip() for i in RLoList]

        self.check_arguments(low_registers=[Ra] + RLoList)

        def STM_func():
            for i in range(len(RLoList)):
                for j in range(4):
                    self.memory[self.register[Ra] + 4*i + j] = ((self.register[RLoList[i]] >> (8 * j)) & 0xFF)
            self.register[Ra] += 4*len(RLoList)

        return STM_func