def write_worksheets(workbook, data_list, result_info_key, identifier_keys):
    """Writes rest of the worksheets to workbook.

    Args:
        workbook: workbook to write into
        data_list: Analytics API data as a list of dicts
        result_info_key: the key in api_data dicts that contains the data results
        identifier_keys: the list of keys used as requested identifiers
                         (address, zipcode, block_id, etc)
    """

    # we can use the first item to figure out the worksheet keys
    worksheet_keys = get_worksheet_keys(data_list[0], result_info_key)

    for key in worksheet_keys:

        title = key.split('/')[1]

        title = utilities.convert_snake_to_title_case(title)

        title = KEY_TO_WORKSHEET_MAP.get(title, title)

        if key == 'property/nod':
            # the property/nod endpoint needs to be split into two worksheets
            create_property_nod_worksheets(workbook, data_list, result_info_key, identifier_keys)
        else:
            # all other endpoints are written to a single worksheet

            # Maximum 31 characters allowed in sheet title
            worksheet = workbook.create_sheet(title=title[:31])

            processed_data = process_data(key, data_list, result_info_key, identifier_keys)

            write_data(worksheet, processed_data)

    # remove the first, unused empty sheet
    workbook.remove_sheet(workbook.active)