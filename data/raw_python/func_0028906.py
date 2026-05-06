def POP(self, params):
        """
        POP {RPopList}

        Pop from the stack into the list of registers
        List must contain only low registers or PC
        """
        # TODO verify pop order
        # TODO pop list is comma separate, right?
        # TODO what registeres are allowed to POP to? Low Registers and PC
        # TODO need to support ranges, ie {R2, R5-R7}
        # TODO PUSH should reverse the list, not POP
        RPopList = self.get_one_parameter(r'\s*{(.*)}(.*)', params).split(',')
        RPopList.reverse()
        RPopList = [i.strip() for i in RPopList]

        def POP_func():
            for register in RPopList:
                # Get 4 bytes
                value = 0
                for i in range(4):
                    # TODO use memory width instead of constants
                    value |= self.memory[self.register['SP'] + i] << (8 * i)

                self.register[register] = value
                self.register['SP'] += 4

        return POP_func