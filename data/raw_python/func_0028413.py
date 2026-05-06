def process_data(key, data_list, result_info_key, identifier_keys):
    """ Given a key as the endpoint name, pulls the data for that endpoint out
        of the data_list for each address, processes the data into a more
        excel-friendly format and returns that data.

        Args:
            key: the endpoint name of the data to process
            data_list: the main data list to take the data from
            result_info_key: the key in api_data dicts that contains the data results
            identifier_keys: the list of keys used as requested identifiers
                             (address, zipcode, block_id, etc)

        Returns:
            A list of dicts (rows) to be written to a worksheet
    """
    master_data = []

    for item_data in data_list:
        data = item_data[key]

        if data is None:
            current_item_data = {}
        else:
            if key == 'property/value':
                current_item_data = data['value']

            elif key == 'property/details':
                top_level_keys = ['property', 'assessment']
                current_item_data = flatten_top_level_keys(data, top_level_keys)

            elif key == 'property/school':
                current_item_data = data['school']

                school_list = []
                for school_type_key in current_item_data:
                    schools = current_item_data[school_type_key]
                    for school in schools:
                        school['school_type'] = school_type_key
                        school['school_address'] = school['address']
                        school['school_zipcode'] = school['zipcode']
                        school_list.append(school)

                current_item_data = school_list

            elif key == 'property/value_forecast':
                current_item_data = {}
                for month_key in data:
                    current_item_data[month_key] = data[month_key]['value']

            elif key in ['property/value_within_block', 'property/rental_value_within_block']:
                current_item_data = flatten_top_level_keys(data, [
                    'housecanary_value_percentile_range',
                    'housecanary_value_sqft_percentile_range',
                    'client_value_percentile_range',
                    'client_value_sqft_percentile_range'
                ])

            elif key in ['property/zip_details', 'zip/details']:
                top_level_keys = ['multi_family', 'single_family']
                current_item_data = flatten_top_level_keys(data, top_level_keys)

            else:
                current_item_data = data

        if isinstance(current_item_data, dict):
            _set_identifier_fields(current_item_data, item_data, result_info_key, identifier_keys)

            master_data.append(current_item_data)
        else:
            # it's a list
            for item in current_item_data:
                _set_identifier_fields(item, item_data, result_info_key, identifier_keys)

            master_data.extend(current_item_data)

    return master_data