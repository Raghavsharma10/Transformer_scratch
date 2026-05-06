def DeriveDataLookupAll(data, configFields, db, histObj={}):
    """
    Return a record after performing derive rules for all fields, based on config

    :param dict data: single record (dictionary) to which cleaning rules should be applied
    :param dict configFields: "fields" object from DWM config (see DataDictionary)
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param dict histObj: History object to which changes should be appended
    """

    def checkDeriveOptions(option, derive_set_config):
        """
        Check derive option is exist into options list and return relevant flag.
        :param option: drive options value
        :param derive_set_config: options list
        :return: boolean True or False based on option exist into options list
        """

        return option in derive_set_config

    for field in configFields.keys():

        if field in data.keys():

            fieldVal = data[field]

            fieldValNew = fieldVal

            for deriveSet in configFields[field]['derive'].keys():

                deriveSetConfig = configFields[field]['derive'][deriveSet]

                checkMatch = False

                if set.issubset(set(deriveSetConfig['fieldSet']), data.keys()):

                    deriveInput = {}

                    # sorting here to ensure subdocument match from query
                    for val in deriveSetConfig['fieldSet']:
                        deriveInput[val] = data[val]

                    if deriveSetConfig['type']=='deriveValue':

                        fieldValNew, histObj, checkMatch = DeriveDataLookup(fieldName=field, db=db, deriveInput=deriveInput, overwrite=checkDeriveOptions('overwrite', deriveSetConfig["options"]), fieldVal=fieldVal, histObj=histObj, blankIfNoMatch=checkDeriveOptions('blankIfNoMatch', deriveSetConfig["options"]))

                    elif deriveSetConfig['type']=='copyValue':

                        fieldValNew, histObj, checkMatch = DeriveDataCopyValue(fieldName=field, deriveInput=deriveInput, overwrite=checkDeriveOptions('overwrite', deriveSetConfig["options"]), fieldVal=fieldVal, histObj=histObj)

                    elif deriveSetConfig['type']=='deriveRegex':

                        fieldValNew, histObj, checkMatch = DeriveDataRegex(fieldName=field, db=db, deriveInput=deriveInput, overwrite=checkDeriveOptions('overwrite', deriveSetConfig["options"]), fieldVal=fieldVal, histObj=histObj, blankIfNoMatch=checkDeriveOptions('blankIfNoMatch', deriveSetConfig["options"]))

                    elif deriveSetConfig['type']=='deriveIncludes':

                        fieldValNew, histObj, checkMatch = IncludesLookup(fieldVal=data[field], lookupType='deriveIncludes', deriveFieldName=deriveSetConfig['fieldSet'][0], deriveInput=deriveInput,  db=db, fieldName=field, histObj=histObj, overwrite=checkDeriveOptions('overwrite', deriveSetConfig["options"]), blankIfNoMatch=checkDeriveOptions('blankIfNoMatch', deriveSetConfig["options"]))

                if checkMatch or fieldValNew!=fieldVal:
                    data[field] = fieldValNew

                    break

    return data, histObj