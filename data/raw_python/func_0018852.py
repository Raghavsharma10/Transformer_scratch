def get_controlfileheader(
        model: Union[str, 'modeltools.Model'],
        parameterstep: timetools.PeriodConstrArg = None,
        simulationstep: timetools.PeriodConstrArg = None) -> str:
    """Return the header of a regular or auxiliary parameter control file.

    The header contains the default coding information, the import command
    for the given model and the actual parameter and simulation step sizes.

    The first example shows that, if you pass the model argument as a
    string, you have to take care that this string makes sense:

    >>> from hydpy.core.parametertools import get_controlfileheader, Parameter
    >>> from hydpy import Period, prepare_model, pub, Timegrids, Timegrid
    >>> print(get_controlfileheader(model='no model class',
    ...                          parameterstep='-1h',
    ...                          simulationstep=Period('1h')))
    # -*- coding: utf-8 -*-
    <BLANKLINE>
    from hydpy.models.no model class import *
    <BLANKLINE>
    simulationstep('1h')
    parameterstep('-1h')
    <BLANKLINE>
    <BLANKLINE>

    The second example shows the saver option to pass the proper model
    object.  It also shows that function |get_controlfileheader| tries
    to gain the parameter and simulation step sizes from the global
    |Timegrids| object contained in the module |pub| when necessary:

    >>> model = prepare_model('lland_v1')
    >>> _ = Parameter.parameterstep('1d')
    >>> pub.timegrids = '2000.01.01', '2001.01.01', '1h'
    >>> print(get_controlfileheader(model=model))
    # -*- coding: utf-8 -*-
    <BLANKLINE>
    from hydpy.models.lland_v1 import *
    <BLANKLINE>
    simulationstep('1h')
    parameterstep('1d')
    <BLANKLINE>
    <BLANKLINE>
    """
    with Parameter.parameterstep(parameterstep):
        if simulationstep is None:
            simulationstep = Parameter.simulationstep
        else:
            simulationstep = timetools.Period(simulationstep)
        return (f"# -*- coding: utf-8 -*-\n\n"
                f"from hydpy.models.{model} import *\n\n"
                f"simulationstep('{simulationstep}')\n"
                f"parameterstep('{Parameter.parameterstep}')\n\n")