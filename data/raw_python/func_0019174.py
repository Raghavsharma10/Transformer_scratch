def save_controls(self, parameterstep=None, simulationstep=None,
                      auxfiler=None):
        """Call method |Elements.save_controls| of the |Elements| object
        currently handled by the |HydPy| object.

        We use the `LahnH` example project to demonstrate how to write
        a complete set parameter control files.  For convenience, we let
        function |prepare_full_example_2| prepare a fully functional
        |HydPy| object, handling seven |Element| objects controlling
        four |hland_v1| and three |hstream_v1| application models:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()

        At first, there is only one control subfolder named "default",
        containing the seven control files used in the step above:

        >>> import os
        >>> with TestIO():
        ...     os.listdir('LahnH/control')
        ['default']

        Next, we use the |ControlManager| to create a new directory
        and dump all control file into it:

        >>> with TestIO():
        ...     pub.controlmanager.currentdir = 'newdir'
        ...     hp.save_controls()
        ...     sorted(os.listdir('LahnH/control'))
        ['default', 'newdir']

        We focus our examples on the (smaller) control files of
        application model |hstream_v1|.  The values of parameter
        |hstream_control.Lag| and |hstream_control.Damp| for the
        river channel connecting the outlets of subcatchment `lahn_1`
        and `lahn_2` are 0.583 days and 0.0, respectively:

        >>> model = hp.elements.stream_lahn_1_lahn_2.model
        >>> model.parameters.control
        lag(0.583)
        damp(0.0)

        The corresponding written control file defines the same values:

        >>> dir_ = 'LahnH/control/newdir/'
        >>> with TestIO():
        ...     with open(dir_ + 'stream_lahn_1_lahn_2.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1d')
        parameterstep('1d')
        <BLANKLINE>
        lag(0.583)
        damp(0.0)
        <BLANKLINE>

        Its name equals the element name and the time step information
        is taken for the |Timegrid| object available via |pub|:

        >>> pub.timegrids.stepsize
        Period('1d')

        Use the |Auxfiler| class To avoid redefining the same parameter
        values in multiple control files.  Here, we prepare an |Auxfiler|
        object which handles the two parameters of the model discussed
        above:

        >>> from hydpy import Auxfiler
        >>> aux = Auxfiler()
        >>> aux += 'hstream_v1'
        >>> aux.hstream_v1.stream = model.parameters.control.damp
        >>> aux.hstream_v1.stream = model.parameters.control.lag

        When passing the |Auxfiler| object to |HydPy.save_controls|,
        both parameters the control file of element `stream_lahn_1_lahn_2`
        do not define their values on their own, but reference the
        auxiliary file `stream.py` instead:

        >>> with TestIO():
        ...     pub.controlmanager.currentdir = 'newdir'
        ...     hp.save_controls(auxfiler=aux)
        ...     with open(dir_ + 'stream_lahn_1_lahn_2.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1d')
        parameterstep('1d')
        <BLANKLINE>
        lag(auxfile='stream')
        damp(auxfile='stream')
        <BLANKLINE>

        `stream.py` contains the actual value definitions:

        >>> with TestIO():
        ...     with open(dir_ + 'stream.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1d')
        parameterstep('1d')
        <BLANKLINE>
        damp(0.0)
        lag(0.583)
        <BLANKLINE>

        The |hstream_v1| model of element `stream_lahn_2_lahn_3` defines
        the same value for parameter |hstream_control.Damp| but a different
        one for parameter |hstream_control.Lag|.  Hence, only
        |hstream_control.Damp| can reference control file `stream.py`
        without distorting data:

        >>> with TestIO():
        ...     with open(dir_ + 'stream_lahn_2_lahn_3.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1d')
        parameterstep('1d')
        <BLANKLINE>
        lag(0.417)
        damp(auxfile='stream')
        <BLANKLINE>

        Another option is to pass alternative step size information.
        The `simulationstep` information, which is not really required
        in control files but useful for testing them, has no impact
        on the written data.  However, passing an alternative
        `parameterstep` information changes the written values of
        time dependent parameters both in the primary and the auxiliary
        control files, as to be expected:

        >>> with TestIO():
        ...     pub.controlmanager.currentdir = 'newdir'
        ...     hp.save_controls(
        ...         auxfiler=aux, parameterstep='2d', simulationstep='1h')
        ...     with open(dir_ + 'stream_lahn_1_lahn_2.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1h')
        parameterstep('2d')
        <BLANKLINE>
        lag(auxfile='stream')
        damp(auxfile='stream')
        <BLANKLINE>

        >>> with TestIO():
        ...     with open(dir_ + 'stream.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1h')
        parameterstep('2d')
        <BLANKLINE>
        damp(0.0)
        lag(0.2915)
        <BLANKLINE>

        >>> with TestIO():
        ...     with open(dir_ + 'stream_lahn_2_lahn_3.py') as controlfile:
        ...         print(controlfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1h')
        parameterstep('2d')
        <BLANKLINE>
        lag(0.2085)
        damp(auxfile='stream')
        <BLANKLINE>
        """
        self.elements.save_controls(parameterstep=parameterstep,
                                    simulationstep=simulationstep,
                                    auxfiler=auxfiler)