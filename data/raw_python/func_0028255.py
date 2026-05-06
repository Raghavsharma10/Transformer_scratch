def export_analytics_data_to_excel(data, output_file_name, result_info_key, identifier_keys):
    """Creates an Excel file containing data returned by the Analytics API

    Args:
        data: Analytics API data as a list of dicts
        output_file_name: File name for output Excel file (use .xlsx extension).

    """
    workbook = create_excel_workbook(data, result_info_key, identifier_keys)
    workbook.save(output_file_name)
    print('Saved Excel file to {}'.format(output_file_name))