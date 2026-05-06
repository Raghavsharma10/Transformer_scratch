def lookupAll(data, configFields, lookupType, db, histObj={}):
    """
    Return a record after having cleaning rules of specified type applied to all fields in the config

    :param dict data: single record (dictionary) to which cleaning rules should be applied
    :param dict configFields: "fields" object from DWM config (see DataDictionary)
    :param string lookupType: Type of lookup to perform/MongoDB collection name. One of 'genericLookup', 'fieldSpecificLookup', 'normLookup', 'genericRegex', 'fieldSpecificRegex', 'normRegex', 'normIncludes'
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param dict histObj: History object to which changes should be appended
    """

    for field in data.keys():

        if field in configFields.keys() and data[field]!='':

            if lookupType in configFields[field]["lookup"]:

                if lookupType in ['genericLookup', 'fieldSpecificLookup', 'normLookup']:

                    fieldValNew, histObj = DataLookup(fieldVal=data[field], db=db, lookupType=lookupType, fieldName=field, histObj=histObj)

                elif lookupType in ['genericRegex', 'fieldSpecificRegex', 'normRegex']:

                    fieldValNew, histObj = RegexLookup(fieldVal=data[field], db=db, fieldName=field, lookupType=lookupType, histObj=histObj)

                elif lookupType=='normIncludes':

                    fieldValNew, histObj, checkMatch = IncludesLookup(fieldVal=data[field], lookupType='normIncludes', db=db, fieldName=field, histObj=histObj)

                data[field] = fieldValNew

    return data, histObj