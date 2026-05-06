def dwmAll(data, db, configName='', config={}, udfNamespace=__name__, verbose=False):
    """
    Return list of dictionaries after cleaning rules have been applied; optionally with a history record ID appended.

    :param list data: list of dictionaries (records) to which cleaning rules should be applied
    :param MongoClient db: MongoDB connection
    :param string configName: name of configuration to use; will be queried from 'config' collection of MongoDB
    :param OrderedDict config: pre-queried config dict
    :param namespace udfNamespace: namespace of current working script; must be passed if using user-defined functions
    :param bool verbose: use tqdm to display progress of cleaning records
    """

    if config=={} and configName=='':
        raise Exception("Please either specify configName or pass a config")

    if config!={} and configName!='':
        raise Exception("Please either specify configName or pass a config")

    if config=={}:
        configColl = db['config']

        config = configColl.find_one({"configName": configName})

        if not config:
            raise Exception("configName '" + configName + "' not found in collection 'config'")

    writeContactHistory = config["history"]["writeContactHistory"]
    returnHistoryId = config["history"]["returnHistoryId"]
    returnHistoryField = config["history"]["returnHistoryField"]
    histIdField = config["history"]["histIdField"]

    for field in config["fields"]:

        config["fields"][field]["derive"] = OrderedDict(sorted(config["fields"][field]["derive"].items()))

    for position in config["userDefinedFunctions"]:

        config["userDefinedFunctions"][position] = OrderedDict(sorted(config["userDefinedFunctions"][position].items()))

    if verbose:
        for row in tqdm(data):
            row, historyId = dwmOne(data=row, db=db, config=config, writeContactHistory=writeContactHistory, returnHistoryId=returnHistoryId, histIdField=histIdField, udfNamespace=udfNamespace)
            if returnHistoryId and writeContactHistory:
                row[returnHistoryField] = historyId
    else:
        for row in data:
            row, historyId = dwmOne(data=row, db=db, config=config, writeContactHistory=writeContactHistory, returnHistoryId=returnHistoryId, histIdField=histIdField, udfNamespace=udfNamespace)
            if returnHistoryId and writeContactHistory:
                row[returnHistoryField] = historyId

    return data