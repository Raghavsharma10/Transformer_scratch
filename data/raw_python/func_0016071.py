def get_code(results):
    """Determines the exit status code to be returned from a script by
    inspecting the results returned from validating file(s).
    Status codes are binary OR'd together, so exit codes can communicate
    multiple error conditions.

    """
    status = EXIT_SUCCESS

    for file_result in results:
        error = any(object_result.errors for object_result in file_result.object_results)

        fatal = file_result.fatal

        if error:
            status |= EXIT_SCHEMA_INVALID
        if fatal:
            status |= EXIT_VALIDATION_ERROR

    return status