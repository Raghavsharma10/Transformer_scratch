def file_comparison(files0, files1):
    """Compares two dictionaries of files returning their difference.

        {'created_files': [<files in files1 and not in files0>],
         'deleted_files': [<files in files0 and not in files1>],
         'modified_files': [<files in both files0 and files1 but different>]}

    """
    comparison = {'created_files': [],
                  'deleted_files': [],
                  'modified_files': []}

    for path, sha1 in files1.items():
        if path in files0:
            if sha1 != files0[path]:
                comparison['modified_files'].append(
                    {'path': path,
                     'original_sha1': files0[path],
                     'sha1': sha1})
        else:
            comparison['created_files'].append({'path': path,
                                                'sha1': sha1})
    for path, sha1 in files0.items():
        if path not in files1:
            comparison['deleted_files'].append({'path': path,
                                                'original_sha1': files0[path]})

    return comparison