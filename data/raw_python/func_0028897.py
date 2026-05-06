def invoke(awsclient, function_name, payload, invocation_type=None,
           alias_name=ALIAS_NAME, version=None, outfile=None):
    """Send a ping request to a lambda function.

    :param awsclient:
    :param function_name:
    :param payload:
    :param invocation_type:
    :param alias_name:
    :param version:
    :param outfile: write response to file
    :return: ping response payload
    """
    log.debug('invoking lambda function: %s', function_name)
    client_lambda = awsclient.get_client('lambda')
    if invocation_type is None:
        invocation_type = 'RequestResponse'
    if payload.startswith('file://'):
        log.debug('reading payload from file: %s' % payload)
        with open(payload[7:], 'r') as pfile:
            payload = pfile.read()

    if version:
        response = client_lambda.invoke(
            FunctionName=function_name,
            InvocationType=invocation_type,
            Payload=payload,
            Qualifier=version
        )
    else:
        response = client_lambda.invoke(
            FunctionName=function_name,
            InvocationType=invocation_type,
            Payload=payload,
            Qualifier=alias_name
        )

    results = response['Payload'].read()  # payload is a 'StreamingBody'
    log.debug('invoke completed')
    # write to file
    if outfile:
        with open(outfile, 'w') as ofile:
            ofile.write(str(results))
            ofile.flush()
        return
    else:
        return results