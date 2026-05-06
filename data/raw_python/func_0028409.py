def create_property_nod_worksheets(workbook, data_list, result_info_key, identifier_keys):
    """Creates two worksheets out of the property/nod data because the data
       doesn't come flat enough to make sense on one sheet.

       Args:
            workbook: the main workbook to add the sheets to
            data_list: the main list of data
            result_info_key: the key in api_data dicts that contains the data results
                             Should always be 'address_info' for property/nod
            identifier_keys: the list of keys used as requested identifiers
                            (address, zipcode, city, state, etc)
    """
    nod_details_list = []
    nod_default_history_list = []

    for prop_data in data_list:
        nod_data = prop_data['property/nod']

        if nod_data is None:
            nod_data = {}

        default_history_data = nod_data.pop('default_history', [])

        _set_identifier_fields(nod_data, prop_data, result_info_key, identifier_keys)

        nod_details_list.append(nod_data)

        for item in default_history_data:
            _set_identifier_fields(item, prop_data, result_info_key, identifier_keys)
            nod_default_history_list.append(item)

    worksheet = workbook.create_sheet(title='NOD Details')
    write_data(worksheet, nod_details_list)

    worksheet = workbook.create_sheet(title='NOD Default History')
    write_data(worksheet, nod_default_history_list)