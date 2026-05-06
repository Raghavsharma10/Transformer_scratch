def functions(context):
    """ Manage AWS Lambda functions """
    # find lambder.json in CWD
    config_file = "./lambder.json"
    if os.path.isfile(config_file):
        context.obj = FunctionConfig(config_file)
    pass