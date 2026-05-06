def dwmOne(data, db, config, writeContactHistory=True, returnHistoryId=True, histIdField={"name": "emailAddress", "value": "emailAddress"}, udfNamespace=__name__):
    """
    Return a single dictionary (record) after cleaning rules have been applied; optionally insert history record to collection 'contactHistory'

    :param dict data: single record (dictionary) to which cleaning rules should be applied
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param dict config: DWM configuration (see DataDictionary)
    :param bool writeContactHistory: Write field-level change history to collection 'contactHistory'
    :param bool returnHistoryId: If writeContactHistory, return '_id' of history record
    :param dict histIdField: Name of identifier for history record: {"name": "emailAddress", "value": "emailAddress"}
    :param namespace udfNamespace: namespace of current working script; must be passed if using user-defined functions
    """

    # setup history collector
    history = {}

    # get user-defined function config
    udFun = config['userDefinedFunctions']

    ## Get runtime field configuration
    fieldConfig = config['fields']

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeGenericValidation", namespace=udfNamespace)

    # Run generic validation lookup
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='genericLookup', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeGenericRegex", namespace=udfNamespace)

    # Run generic validation regex
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='genericRegex', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeFieldSpecificValidation", namespace=udfNamespace)

    # Run field-specific validation lookup
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='fieldSpecificLookup', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeFieldSpecificRegex", namespace=udfNamespace)

    # Run field-specific validation regex
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='fieldSpecificRegex', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeNormalization", namespace=udfNamespace)

    # Run normalization lookup
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='normLookup', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeNormalizationRegex", namespace=udfNamespace)

    # Run normalization regex
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='normRegex', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeNormalizationIncludes", namespace=udfNamespace)

    # Run normalization includes
    data, history = lookupAll(data=data, configFields=fieldConfig, lookupType='normIncludes', db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="beforeDeriveData", namespace=udfNamespace)

    # Fill gaps / refresh derived data
    data, history = DeriveDataLookupAll(data=data, configFields=fieldConfig, db=db, histObj=history)

    ## Run user-defined functions
    data, history = _RunUserDefinedFunctions_(config=config, data=data, histObj=history, position="afterProcessing", namespace=udfNamespace)

    # check if need to write contact change history
    if writeContactHistory:
        history['timestamp'] = int(time.time())
        history[histIdField['name']] = data[histIdField['value']]
        history['configName'] = config['configName']

        # Set _current value for most recent contact
        history['_current'] = 0
       
        # Increment all _current
        db['contactHistory'].update({histIdField['name']: data[histIdField['value']]}, {'$inc': {'_current': 1}}, multi=True)

        # Insert into DB
        historyId = db['contactHistory'].insert_one(history).inserted_id

    if writeContactHistory and returnHistoryId:
        return data, historyId
    else:
        return data, None