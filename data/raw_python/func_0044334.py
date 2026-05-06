def get_base_url(config, args):
    """
    Get the API base url. Try Terraform state first, then
    :py:class:`~.AWSInfo`.

    :param config: configuration
    :type config: :py:class:`~.Config`
    :param args: command line arguments
    :type args: :py:class:`argparse.Namespace`
    :return: API base URL
    :rtype: str
    """
    try:
        logger.debug('Trying to get Terraform base_url output')
        runner = TerraformRunner(config, args.tf_path)
        outputs = runner._get_outputs()
        base_url = outputs['base_url']
        logger.debug("Terraform base_url output: '%s'", base_url)
    except Exception:
        logger.info('Unable to find API base_url from Terraform state; '
                    'querying AWS.', exc_info=1)
        aws = AWSInfo(config)
        base_url = aws.get_api_base_url()
        logger.debug("AWS api_base_url: '%s'", base_url)
    if not base_url.endswith('/'):
        base_url += '/'
    return base_url