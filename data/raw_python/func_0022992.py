def add_program(self, name=None):
        """Create a program and add it to this MultiProgram.
        
        It is the caller's responsibility to keep a reference to the returned 
        program.
        
        The *name* must be unique, but is otherwise arbitrary and used for 
        debugging purposes.
        """
        if name is None:
            name = 'program' + str(self._next_prog_id)
            self._next_prog_id += 1
                
        if name in self._programs:
            raise KeyError("Program named '%s' already exists." % name)
        
        # create a program and update it to look like the rest
        prog = ModularProgram(self._vcode, self._fcode)
        for key, val in self._set_items.items():
            prog[key] = val
        self.frag._new_program(prog)
        self.vert._new_program(prog)
        
        self._programs[name] = prog
        return prog