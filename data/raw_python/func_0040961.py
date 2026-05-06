def excel_to_dict(excel_filepath, encapsulate_filepath=False, **kwargs):
    """
    Turn excel into dict.
    Args:
        :excel_filepath: path to excel file to turn into dict.
        :limits: path to csv file to turn into dict
    """
    result = {}
    try:
        callbacks = {'to_dictlist': excel_todictlist}  # Default callback
        callbacks.update(kwargs.get('alt_callbacks', {}))

        # Retrieve excel data as dict of sheets lists
        excel_data = callbacks.get('to_dictlist')(excel_filepath, **kwargs)
        for sheet in excel_data.keys():
            try:
                kwargs['rows'] = excel_data.get(sheet, [])
                result[sheet] = csv_to_dict(excel_filepath, **kwargs)
            except Exception as ex:
                logger.error('Fail to parse sheet {} - {}'.format(sheet, ex))
                result[sheet] = []
                continue

        if encapsulate_filepath:
            result = {excel_filepath: result}

    except Exception as ex:
        msg = 'Fail transform excel to dict - {}'.format(ex)
        logger.error(msg, excel_filepath=excel_filepath)

    return result