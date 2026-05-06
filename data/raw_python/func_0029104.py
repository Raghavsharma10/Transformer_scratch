def deploy(awsclient, applicationName, deploymentGroupName,
           deploymentConfigName, bucket, bundlefile):
    """Upload bundle and deploy to deployment group.
    This includes the bundle-action.

    :param applicationName:
    :param deploymentGroupName:
    :param deploymentConfigName:
    :param bucket:
    :param bundlefile:
    :return: deploymentId from create_deployment
    """
    etag, version = upload_file_to_s3(awsclient, bucket,
                                      _build_bundle_key(applicationName),
                                      bundlefile)

    client_codedeploy = awsclient.get_client('codedeploy')
    response = client_codedeploy.create_deployment(
        applicationName=applicationName,
        deploymentGroupName=deploymentGroupName,
        revision={
            'revisionType': 'S3',
            's3Location': {
                'bucket': bucket,
                'key': _build_bundle_key(applicationName),
                'bundleType': 'tgz',
                'eTag': etag,
                'version': version,
            },
        },
        deploymentConfigName=deploymentConfigName,
        description='deploy with tenkai',
        ignoreApplicationStopFailures=True
    )

    log.info(
        "Deployment: {} -> URL: https://{}.console.aws.amazon.com/codedeploy/home?region={}#/deployments/{}".format(
            Fore.MAGENTA + response['deploymentId'] + Fore.RESET,
            client_codedeploy.meta.region_name,
            client_codedeploy.meta.region_name,
            response['deploymentId'],
        ))

    return response['deploymentId']