def list_functions(awsclient):
    """List the deployed lambda functions and print configuration.

    :return: exit_code
    """
    client_lambda = awsclient.get_client('lambda')
    response = client_lambda.list_functions()
    for function in response['Functions']:
        log.info(function['FunctionName'])
        log.info('\t' 'Memory: ' + str(function['MemorySize']))
        log.info('\t' 'Timeout: ' + str(function['Timeout']))
        log.info('\t' 'Role: ' + str(function['Role']))
        log.info('\t' 'Current Version: ' + str(function['Version']))
        log.info('\t' 'Last Modified: ' + str(function['LastModified']))
        log.info('\t' 'CodeSha256: ' + str(function['CodeSha256']))

        log.info('\n')
    return 0