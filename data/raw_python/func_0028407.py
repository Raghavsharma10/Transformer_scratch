def get_excel_workbook(api_data, result_info_key, identifier_keys):
    """Generates an Excel workbook object given api_data returned by the Analytics API

    Args:
        api_data: Analytics API data as a list of dicts (one per identifier)
        result_info_key: the key in api_data dicts that contains the data results
        identifier_keys: the list of keys used as requested identifiers
                         (address, zipcode, block_id, etc)

    Returns:
        raw excel file data
    """

    cleaned_data = []

    for item_data in api_data:
        result_info = item_data.pop(result_info_key, {})

        cleaned_item_data = {}

        if 'meta' in item_data:
            meta = item_data.pop('meta')
            cleaned_item_data['meta'] = meta

        for key in item_data:
            cleaned_item_data[key] = item_data[key]['result']

        cleaned_item_data[result_info_key] = result_info

        cleaned_data.append(cleaned_item_data)

    data_list = copy.deepcopy(cleaned_data)

    workbook = openpyxl.Workbook()

    write_worksheets(workbook, data_list, result_info_key, identifier_keys)

    return workbook