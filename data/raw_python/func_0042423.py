def assign_operation_ids(spec, operids):
    """ used to assign caller provided operationId values into a spec """

    empty_dict = {}

    for path_name, path_data in six.iteritems(spec['paths']):
        for method, method_data in six.iteritems(path_data):
            oper_id = operids.get(path_name, empty_dict).get(method)
            if oper_id:
                method_data['operationId'] = oper_id