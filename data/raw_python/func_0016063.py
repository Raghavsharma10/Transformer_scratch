def print_file_results(file_result):
    """Print the results of validating a file.

    Args:
        file_result: A FileValidationResults instance.

    """
    print_results_header(file_result.filepath, file_result.is_valid)

    for object_result in file_result.object_results:
        if object_result.warnings:
            print_warning_results(object_result, 1)
        if object_result.errors:
            print_schema_results(object_result, 1)

    if file_result.fatal:
        print_fatal_results(file_result.fatal, 1)