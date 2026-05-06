def save_controls(self, filepath: Optional[str] = None,
                      parameterstep: timetools.PeriodConstrArg = None,
                      simulationstep: timetools.PeriodConstrArg = None,
                      auxfiler: 'auxfiletools.Auxfiler' = None):
        """Write the control parameters to file.

        Usually, a control file consists of a header (see the documentation
        on the method |get_controlfileheader|) and the string representations
        of the individual |Parameter| objects handled by the `control`
        |SubParameters| object.

        The main functionality of method |Parameters.save_controls| is
        demonstrated in the documentation on the method |HydPy.save_controls|
        of class |HydPy|, which one would apply to write the parameter
        information of complete *HydPy* projects.  However, to call
        |Parameters.save_controls| on individual |Parameters| objects
        offers the advantage to choose an arbitrary file path, as shown
        in the following example:

        >>> from hydpy.models.hstream_v1 import *
        >>> parameterstep('1d')
        >>> simulationstep('1h')
        >>> lag(1.0)
        >>> damp(0.5)

        >>> from hydpy import Open
        >>> with Open():
        ...     model.parameters.save_controls('otherdir/otherfile.py')
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        otherdir/otherfile.py
        -------------------------------------
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy.models.hstream_v1 import *
        <BLANKLINE>
        simulationstep('1h')
        parameterstep('1d')
        <BLANKLINE>
        lag(1.0)
        damp(0.5)
        <BLANKLINE>
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Without a given file path and a proper project configuration,
        method |Parameters.save_controls| raises the following error:

        >>> model.parameters.save_controls()
        Traceback (most recent call last):
        ...
        RuntimeError: To save the control parameters of a model to a file, \
its filename must be known.  This can be done, by passing a filename to \
function `save_controls` directly.  But in complete HydPy applications, \
it is usally assumed to be consistent with the name of the element \
handling the model.
        """
        if self.control:
            variable2auxfile = getattr(auxfiler, str(self.model), None)
            lines = [get_controlfileheader(
                self.model, parameterstep, simulationstep)]
            with Parameter.parameterstep(parameterstep):
                for par in self.control:
                    if variable2auxfile:
                        auxfilename = variable2auxfile.get_filename(par)
                        if auxfilename:
                            lines.append(
                                f"{par.name}(auxfile='{auxfilename}')\n")
                            continue
                    lines.append(repr(par) + '\n')
            text = ''.join(lines)
            if filepath:
                with open(filepath, mode='w', encoding='utf-8') as controlfile:
                    controlfile.write(text)
            else:
                filename = objecttools.devicename(self)
                if filename == '?':
                    raise RuntimeError(
                        'To save the control parameters of a model to a file, '
                        'its filename must be known.  This can be done, by '
                        'passing a filename to function `save_controls` '
                        'directly.  But in complete HydPy applications, it is '
                        'usally assumed to be consistent with the name of the '
                        'element handling the model.')
                hydpy.pub.controlmanager.save_file(filename, text)