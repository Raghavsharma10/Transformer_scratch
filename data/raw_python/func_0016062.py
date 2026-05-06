def print_object_results(obj_result):
    """Print the results of validating an object.

    Args:
        obj_result: An ObjectValidationResults instance.

    """
    print_results_header(obj_result.object_id, obj_result.is_valid)

    if obj_result.warnings:
        print_warning_results(obj_result, 1)
    if obj_result.errors:
        print_schema_results(obj_result, 1)