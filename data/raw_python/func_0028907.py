def PUSH(self, params):
        """
        PUSH {RPushList}

        Push to the stack from a list of registers
        List must contain only low registers or LR
        """
        # TODO what registers are allowed to PUSH to? Low registers and LR
        # TODO PUSH should reverse the list, not POP
        RPushList = self.get_one_parameter(r'\s*{(.*)}(.*)', params).split(',')
        RPushList = [i.strip() for i in RPushList]
        # TODO should we make sure the register exists? probably not

        def PUSH_func():
            for register in RPushList:
                self.register['SP'] -= 4

                for i in range(4):
                    # TODO is this the same as with POP?
                    self.memory[self.register['SP'] + i] = ((self.register[register] >> (8 * i)) & 0xFF)

        return PUSH_func