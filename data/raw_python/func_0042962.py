def _RunUserDefinedFunctions_(config, data, histObj, position, namespace=__name__):
    """
    Return a single updated data record and history object after running user-defined functions

    :param dict config: DWM configuration (see DataDictionary)
    :param dict data: single record (dictionary) to which user-defined functions should be applied
    :param dict histObj: History object to which changes should be appended
    :param string position: position name of which function set from config should be run
    :param namespace: namespace of current working script; must be passed if using user-defined functions
    """

    udfConfig = config['userDefinedFunctions']

    if position in udfConfig:

        posConfig = udfConfig[position]

        for udf in posConfig.keys():

            posConfigUDF = posConfig[udf]

            data, histObj = getattr(sys.modules[namespace], posConfigUDF)(data=data, histObj=histObj)

    return data, histObj