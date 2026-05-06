def load_conditions(self, filename=None):
        """Read the initial conditions from a file and assign them to the
        respective |StateSequence| and/or |LogSequence| objects handled by
        the actual |Sequences| object.

        If no filename or dirname is passed, the ones defined by the
        |ConditionManager| stored in module |pub| are used.
        """
        if self.hasconditions:
            if not filename:
                filename = self._conditiondefaultfilename
            namespace = locals()
            for seq in self.conditionsequences:
                namespace[seq.name] = seq
            namespace['model'] = self
            code = hydpy.pub.conditionmanager.load_file(filename)
            try:
                # ToDo: raises an escape sequence deprecation sometimes
                # ToDo: use runpy instead?
                # ToDo: Move functionality to filetools.py?
                exec(code)
            except BaseException:
                objecttools.augment_excmessage(
                    'While trying to gather initial conditions of element %s'
                    % objecttools.devicename(self))