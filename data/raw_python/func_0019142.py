def controlcheck(controldir='default', projectdir=None, controlfile=None):
    """Define the corresponding control file within a condition file.

    Function |controlcheck| serves similar purposes as function
    |parameterstep|.  It is the reason why one can interactively
    access the state and/or the log sequences within condition files
    as `land_dill.py` of the example project `LahnH`.  It is called
    `controlcheck` due to its implicite feature to check upon the execution
    of the condition file if eventual specifications within both files
    disagree.  The following test, where we write a number of soil moisture
    values (|hland_states.SM|) into condition file `land_dill.py` which
    does not agree with the number of hydrological response units
    (|hland_control.NmbZones|) defined in control file `land_dill.py`,
    verifies that this actually works within a new Python process:

    >>> from hydpy.core.examples import prepare_full_example_1
    >>> prepare_full_example_1()

    >>> import os, subprocess
    >>> from hydpy import TestIO
    >>> cwd = os.path.join('LahnH', 'conditions', 'init_1996_01_01')
    >>> with TestIO():
    ...     os.chdir(cwd)
    ...     with open('land_dill.py') as file_:
    ...         lines = file_.readlines()
    ...     lines[10:12] = 'sm(185.13164, 181.18755)', ''
    ...     with open('land_dill.py', 'w') as file_:
    ...         _ = file_.write('\\n'.join(lines))
    ...     result = subprocess.run(
    ...         'python land_dill.py',
    ...         stdout=subprocess.PIPE,
    ...         stderr=subprocess.PIPE,
    ...         universal_newlines=True,
    ...         shell=True)
    >>> print(result.stderr.split('ValueError:')[-1].strip())
    While trying to set the value(s) of variable `sm`, the following error \
occurred: While trying to convert the value(s) `(185.13164, 181.18755)` to \
a numpy ndarray with shape `(12,)` and type `float`, the following error \
occurred: could not broadcast input array from shape (2) into shape (12)

    With a little trick, we can fake to be "inside" condition file
    `land_dill.py`.  Calling |controlcheck| then e.g. prepares the shape
    of sequence |hland_states.Ic| as specified by the value of parameter
    |hland_control.NmbZones| given in the corresponding control file:

    >>> from hydpy.models.hland_v1 import *
    >>> __file__ = 'land_dill.py'   # ToDo: undo?
    >>> with TestIO():
    ...     os.chdir(cwd)
    ...     controlcheck()
    >>> ic.shape
    (12,)

    In the above example, the standard names for the project directory
    (the one containing the executed condition file) and the control
    directory (`default`) are used.  The following example shows how
    to change them:

    >>> del model
    >>> with TestIO():   # doctest: +ELLIPSIS
    ...     os.chdir(cwd)
    ...     controlcheck(projectdir='somewhere', controldir='nowhere')
    Traceback (most recent call last):
    ...
    FileNotFoundError: While trying to load the control file \
`...hydpy...tests...iotesting...control...nowhere...land_dill.py`, the \
following error occurred: [Errno 2] No such file or directory: '...land_dill.py'

    Note that the functionalities of function |controlcheck| are disabled
    when there is already a `model` variable in the namespace, which is
    the case when a condition file is executed within the context of a
    complete HydPy project.
    """
    namespace = inspect.currentframe().f_back.f_locals
    model = namespace.get('model')
    if model is None:
        if not controlfile:
            controlfile = os.path.split(namespace['__file__'])[-1]
        if projectdir is None:
            projectdir = (
                os.path.split(
                    os.path.split(
                        os.path.split(os.getcwd())[0])[0])[-1])
        dirpath = os.path.abspath(os.path.join(
            '..', '..', '..', projectdir, 'control', controldir))

        class CM(filetools.ControlManager):
            currentpath = dirpath

        model = CM().load_file(filename=controlfile)['model']
        model.parameters.update()
        namespace['model'] = model
        for name in ('states', 'logs'):
            subseqs = getattr(model.sequences, name, None)
            if subseqs is not None:
                for seq in subseqs:
                    namespace[seq.name] = seq