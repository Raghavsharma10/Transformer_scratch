def export_analytics_data_to_csv(data, output_folder, result_info_key, identifier_keys):
    """Creates CSV files containing data returned by the Analytics API.
       Creates one file per requested endpoint and saves it into the
       specified output_folder

    Args:
        data: Analytics API data as a list of dicts
        output_folder: Path to a folder to save the CSV files into
    """
    workbook = create_excel_workbook(data, result_info_key, identifier_keys)

    suffix = '.csv'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for worksheet in workbook.worksheets:
        file_name = utilities.convert_title_to_snake_case(worksheet.title)

        file_path = os.path.join(output_folder, file_name + suffix)

        mode = 'w'
        if sys.version_info[0] < 3:
            mode = 'wb'
        with io.open(file_path, mode) as output_file:
            csv_writer = csv.writer(output_file)
            for row in worksheet.rows:
                csv_writer.writerow([cell.value for cell in row])

    print('Saved CSV files to {}'.format(output_folder))