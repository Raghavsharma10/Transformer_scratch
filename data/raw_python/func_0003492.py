def set_debugger(self, debugger):
        """
        Set debugger to run when an error occurs in published method.

        You can also set debugger by passing `debugger` argument to
        the class constructor.

        :type debugger: {'pdb', 'ipdb', None}
        :arg  debugger: type of debugger.

        """
        if debugger == 'pdb':
            import pdb
            self.debugger = pdb
        elif debugger == 'ipdb':
            import ipdb
            self.debugger = ipdb
        else:
            self.debugger = debugger