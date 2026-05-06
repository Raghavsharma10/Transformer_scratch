def concat_excel_reports(addresses, output_file_name, endpoint, report_type,
                         retry, api_key, api_secret, files_path):
    """Creates an Excel file made up of combining the Value Report or Rental Report Excel
       output for the provided addresses.

    Args:
        addresses: A list of (address, zipcode) tuples
        output_file_name: A file name for the Excel output
        endpoint: One of 'value_report' or 'rental_report'
        report_type: One of 'full' or 'summary'
        retry: optional boolean to retry if rate limit is reached
        api_key: optional API Key
        api_secret: optional API Secret
        files_path: Path to save individual files. If None, don't save files
    """
    # create the master workbook to output
    master_workbook = openpyxl.Workbook()

    if api_key is not None and api_secret is not None:
        client = ApiClient(api_key, api_secret)
    else:
        client = ApiClient()

    errors = []

    # for each address, call the API and load the xlsx content in a workbook.
    for index, addr in enumerate(addresses):
        print('Processing {}'.format(addr[0]))
        result = _get_excel_report(
            client, endpoint, addr[0], addr[1], report_type, retry)

        if not result['success']:
            print('Error retrieving report for {}'.format(addr[0]))
            print(result['content'])
            errors.append({'address': addr[0], 'message': result['content']})
            continue

        orig_wb = openpyxl.load_workbook(filename=io.BytesIO(result['content']))

        _save_individual_file(orig_wb, files_path, addr[0])

        # for each worksheet for this address
        for sheet_name in orig_wb.get_sheet_names():
            # if worksheet doesn't exist in master workbook, create it
            if sheet_name in master_workbook.get_sheet_names():
                master_ws = master_workbook.get_sheet_by_name(sheet_name)
            else:
                master_ws = master_workbook.create_sheet(sheet_name)

            # get all the rows in the address worksheet
            orig_rows = orig_wb.get_sheet_by_name(sheet_name).rows

            if sheet_name == 'Summary' or sheet_name == 'Chart Data':
                _process_non_standard_sheet(master_ws, orig_rows, addr, index)
                continue

            _process_standard_sheet(master_ws, orig_rows, addr, index)

    # remove the first sheet which will be empty
    master_workbook.remove(master_workbook.worksheets[0])

    # if any errors occurred, write them to an "Errors" worksheet
    if len(errors) > 0:
        errors_sheet = master_workbook.create_sheet('Errors')
        for error_idx, error in enumerate(errors):
            errors_sheet.cell(row=error_idx+1, column=1, value=error['address'])
            errors_sheet.cell(row=error_idx+1, column=2, value=error['message'])

    # save the master workbook to output_file_name
    adjust_column_width_workbook(master_workbook)
    output_file_path = os.path.join(files_path, output_file_name)
    master_workbook.save(output_file_path)
    print('Saved output to {}'.format(output_file_path))