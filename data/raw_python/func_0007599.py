def download_metadata(master_token, output_csv, verbose=False, debug=False):
    """
    Output CSV with metadata for a project's downloadable files in Open Humans.

    :param master_token: This field is the master access token for the project.
    :param output_csv: This field is the target csv file to which metadata is
        written.
    :param verbose: This boolean field is the logging level. It's default value
        is False.
    :param debug: This boolean field is the logging level. It's default value
        is False.
    """
    set_log_level(debug, verbose)

    project = OHProject(master_access_token=master_token)

    with open(output_csv, 'w') as f:
        csv_writer = csv.writer(f)
        header = ['project_member_id', 'data_source', 'file_basename',
                  'file_upload_date']
        csv_writer.writerow(header)
        for member_id in project.project_data:
            if not project.project_data[member_id]['data']:
                csv_writer.writerow([member_id, 'NA', 'None', 'NA'])
            else:
                for data_item in project.project_data[member_id]['data']:
                    logging.debug(data_item)
                    csv_writer.writerow([
                        member_id, data_item['source'],
                        data_item['basename'].encode('utf-8'),
                        data_item['created']])