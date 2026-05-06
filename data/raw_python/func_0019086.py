def save(self, parameterstep=None, simulationstep=None):
        """Save all defined auxiliary control files.

        The target path is taken from the |ControlManager| object stored
        in module |pub|.  Hence we initialize one and override its
        |property| `currentpath` with a simple |str| object defining the
        test target path:

        >>> from hydpy import pub
        >>> pub.projectname = 'test'
        >>> from hydpy.core.filetools import ControlManager
        >>> class Test(ControlManager):
        ...     currentpath = 'test_directory'
        >>> pub.controlmanager = Test()

        Normally, the control files would be written to disk, of course.
        But to show (and test) the results in the following doctest,
        file writing is temporarily redirected via |Open|:

        >>> from hydpy import dummies
        >>> from hydpy import Open
        >>> with Open():
        ...     dummies.aux.save(
        ...         parameterstep='1d',
        ...         simulationstep='12h')
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        test_directory/file1.py
        -----------------------------------
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.lland_v1 import *
        <BLANKLINE>
        simulationstep('12h')
        parameterstep('1d')
        <BLANKLINE>
        eqd1(200.0)
        <BLANKLINE>
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        test_directory/file2.py
        -----------------------------------
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.lland_v2 import *
        <BLANKLINE>
        simulationstep('12h')
        parameterstep('1d')
        <BLANKLINE>
        eqd1(200.0)
        eqd2(100.0)
        <BLANKLINE>
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        par = parametertools.Parameter
        for (modelname, var2aux) in self:
            for filename in var2aux.filenames:
                with par.parameterstep(parameterstep), \
                         par.simulationstep(simulationstep):
                    lines = [parametertools.get_controlfileheader(
                        modelname, parameterstep, simulationstep)]
                    for par in getattr(var2aux, filename):
                        lines.append(repr(par) + '\n')
                hydpy.pub.controlmanager.save_file(filename, ''.join(lines))