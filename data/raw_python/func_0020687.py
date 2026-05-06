def validateStringInput(input_key,input_data, read=False):
    """
    To check if a string has the required format. This is only used for POST APIs.
    """
    log = clog.error_log
    func = None
    if '*' in input_data or '%' in input_data:
        func = validationFunctionWildcard.get(input_key)
        if func is None:
            func = searchstr
    elif input_key == 'migration_input' :
        if input_data.find('#') != -1 : func = block
        else : func = dataset
    else:
        if not read:
            func = validationFunction.get(input_key)
            if func is None:
                func = namestr
        else:
            if input_key == 'dataset':
                func = reading_dataset_check
            elif input_key == 'block_name':
                func = reading_block_check
            elif input_key == 'logical_file_name':
                func = reading_lfn_check
            else:
                func = namestr
    try:
        func(input_data)
    except AssertionError as ae:
        serverLog = str(ae) + " key-value pair (%s, %s) cannot pass input checking" %(input_key, input_data)
        #print serverLog
        dbsExceptionHandler("dbsException-invalid-input2", message="Invalid Input Data %s...:  Not Match Required Format" %input_data[:10], \
            logger=log.error, serverError=serverLog)
    return input_data