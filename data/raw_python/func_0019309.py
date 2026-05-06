def save_conditions(self, filename=None):
        """Query the actual conditions of the |StateSequence| and/or
        |LogSequence| objects handled by the actual |Sequences| object and
        write them into a initial condition file.

        If no filename or dirname is passed, the ones defined by the
        |ConditionManager| stored in module |pub| are used.
        """
        if self.hasconditions:
            if filename is None:
                filename = self._conditiondefaultfilename
            con = hydpy.pub.controlmanager
            lines = ['# -*- coding: utf-8 -*-\n\n',
                     'from hydpy.models.%s import *\n\n' % self.model,
                     'controlcheck(projectdir="%s", controldir="%s")\n\n'
                     % (con.projectdir, con.currentdir)]
            for seq in self.conditionsequences:
                lines.append(repr(seq) + '\n')
            hydpy.pub.conditionmanager.save_file(filename, ''.join(lines))