def write_composer_operation_log(filename):
    """
    Writes the composed operation log from featuremonkey's Composer to a json file.
    :param filename:
    :return:
    """
    from featuremonkey.tracing import serializer
    from featuremonkey.tracing.logger import OPERATION_LOG
    ol = copy.deepcopy(OPERATION_LOG)
    ol = serializer.serialize_operation_log(ol)
    with open(filename, 'w+') as operation_log_file:
        operation_log_file.write(json.dumps(ol, indent=4, encoding="utf8"))