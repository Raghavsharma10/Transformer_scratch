def get_api_id(config, args):
    """
    Get the API ID from Terraform, or from AWS if that fails.

    :param config: configuration
    :type config: :py:class:`~.Config`
    :param args: command line arguments
    :type args: :py:class:`argparse.Namespace`
    :return: API Gateway ID
    :rtype: str
    """
    try:
        logger.debug('Trying to get Terraform rest_api_id output')
        runner = TerraformRunner(config, args.tf_path)
        outputs = runner._get_outputs()
        depl_id = outputs['rest_api_id']
        logger.debug("Terraform rest_api_id output: '%s'", depl_id)
    except Exception:
        logger.info('Unable to find API rest_api_id from Terraform state;'
                    ' querying AWS.', exc_info=1)
        aws = AWSInfo(config)
        depl_id = aws.get_api_id()
        logger.debug("AWS API ID: '%s'", depl_id)
    return depl_id