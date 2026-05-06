def sim(self, key, size=None):
        '''
        key: memory address(int) or register name(str)
        size: size of object in bytes
        '''
        project = load_project()
        if key in project.arch.registers:
            if size is None:
                size = project.arch.registers[key][1]
            size *= 8
            s = claripy.BVS("angrdbg_reg_" + str(key), size)
            setattr(self.state.regs, key, s)
            self.symbolics[key] = (s, size)
        elif isinstance(key, int) or isinstance(key, long):
            if size is None:
                size = project.arch.bits
            else:
                size *= 8
            s = claripy.BVS("angrdbg_mem_" + hex(key), size)
            self.state.memory.store(key, s)
            self.symbolics[key] = (s, size)
        elif isinstance(key, claripy.ast.bv.BV):
            key = self.state.solver.eval(key, cast_to=int)
            self.sim(key, size)
        else:
            raise ValueError(
                "key must be a register name or a memory address, not %s" % str(
                    type(key)))
        return key