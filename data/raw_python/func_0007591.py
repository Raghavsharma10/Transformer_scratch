def write_metadata_to_filestream(filedir, filestream,
                                 max_bytes=MAX_FILE_DEFAULT):
    """
    Make metadata file for all files in a directory(helper function)

    :param filedir: This field is the filepath of the directory whose csv
        has to be made.
    :param filestream: This field is a stream for writing to the csv.
    :param max_bytes: This field is the maximum file size to consider. Its
        default value is 128m.
    """
    csv_out = csv.writer(filestream)
    subdirs = [os.path.join(filedir, i) for i in os.listdir(filedir) if
               os.path.isdir(os.path.join(filedir, i))]
    if subdirs:
        logging.info('Making metadata for subdirs of {}'.format(filedir))
        if not all([re.match('^[0-9]{8}$', os.path.basename(d))
                    for d in subdirs]):
            raise ValueError("Subdirs not all project member ID format!")
        csv_out.writerow(['project_member_id', 'filename', 'tags',
                          'description', 'md5', 'creation_date'])
        for subdir in subdirs:
            file_info = characterize_local_files(
                filedir=subdir, max_bytes=max_bytes)
            proj_member_id = os.path.basename(subdir)
            if not file_info:
                csv_out.writerow([proj_member_id, 'None',
                                  'NA', 'NA', 'NA', 'NA'])
                continue
            for filename in file_info:
                csv_out.writerow([proj_member_id,
                                  filename,
                                  ', '.join(file_info[filename]['tags']),
                                  file_info[filename]['description'],
                                  file_info[filename]['md5'],
                                  file_info[filename]['creation_date'],
                                  ])
    else:
        csv_out.writerow(['filename', 'tags',
                          'description', 'md5', 'creation_date'])
        file_info = characterize_local_files(
            filedir=filedir, max_bytes=max_bytes)
        for filename in file_info:
            csv_out.writerow([filename,
                              ', '.join(file_info[filename]['tags']),
                              file_info[filename]['description'],
                              file_info[filename]['md5'],
                              file_info[filename]['creation_date'],
                              ])