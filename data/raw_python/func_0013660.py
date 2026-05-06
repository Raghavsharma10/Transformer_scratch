def create_deployment(applicationName=None, deploymentGroupName=None, revision=None, deploymentConfigName=None, description=None, ignoreApplicationStopFailures=None, targetInstances=None, autoRollbackConfiguration=None, updateOutdatedInstancesOnly=None, fileExistsBehavior=None):
    """
    Deploys an application revision through the specified deployment group.
    See also: AWS API Documentation
    
    
    :example: response = client.create_deployment(
        applicationName='string',
        deploymentGroupName='string',
        revision={
            'revisionType': 'S3'|'GitHub',
            's3Location': {
                'bucket': 'string',
                'key': 'string',
                'bundleType': 'tar'|'tgz'|'zip',
                'version': 'string',
                'eTag': 'string'
            },
            'gitHubLocation': {
                'repository': 'string',
                'commitId': 'string'
            }
        },
        deploymentConfigName='string',
        description='string',
        ignoreApplicationStopFailures=True|False,
        targetInstances={
            'tagFilters': [
                {
                    'Key': 'string',
                    'Value': 'string',
                    'Type': 'KEY_ONLY'|'VALUE_ONLY'|'KEY_AND_VALUE'
                },
            ],
            'autoScalingGroups': [
                'string',
            ]
        },
        autoRollbackConfiguration={
            'enabled': True|False,
            'events': [
                'DEPLOYMENT_FAILURE'|'DEPLOYMENT_STOP_ON_ALARM'|'DEPLOYMENT_STOP_ON_REQUEST',
            ]
        },
        updateOutdatedInstancesOnly=True|False,
        fileExistsBehavior='DISALLOW'|'OVERWRITE'|'RETAIN'
    )
    
    
    :type applicationName: string
    :param applicationName: [REQUIRED]
            The name of an AWS CodeDeploy application associated with the applicable IAM user or AWS account.
            

    :type deploymentGroupName: string
    :param deploymentGroupName: The name of the deployment group.

    :type revision: dict
    :param revision: The type and location of the revision to deploy.
            revisionType (string) --The type of application revision:
            S3: An application revision stored in Amazon S3.
            GitHub: An application revision stored in GitHub.
            s3Location (dict) --Information about the location of application artifacts stored in Amazon S3.
            bucket (string) --The name of the Amazon S3 bucket where the application revision is stored.
            key (string) --The name of the Amazon S3 object that represents the bundled artifacts for the application revision.
            bundleType (string) --The file type of the application revision. Must be one of the following:
            tar: A tar archive file.
            tgz: A compressed tar archive file.
            zip: A zip archive file.
            version (string) --A specific version of the Amazon S3 object that represents the bundled artifacts for the application revision.
            If the version is not specified, the system will use the most recent version by default.
            eTag (string) --The ETag of the Amazon S3 object that represents the bundled artifacts for the application revision.
            If the ETag is not specified as an input parameter, ETag validation of the object will be skipped.
            gitHubLocation (dict) --Information about the location of application artifacts stored in GitHub.
            repository (string) --The GitHub account and repository pair that stores a reference to the commit that represents the bundled artifacts for the application revision.
            Specified as account/repository.
            commitId (string) --The SHA1 commit ID of the GitHub commit that represents the bundled artifacts for the application revision.
            
            

    :type deploymentConfigName: string
    :param deploymentConfigName: The name of a deployment configuration associated with the applicable IAM user or AWS account.
            If not specified, the value configured in the deployment group will be used as the default. If the deployment group does not have a deployment configuration associated with it, then CodeDeployDefault.OneAtATime will be used by default.
            

    :type description: string
    :param description: A comment about the deployment.

    :type ignoreApplicationStopFailures: boolean
    :param ignoreApplicationStopFailures: If set to true, then if the deployment causes the ApplicationStop deployment lifecycle event to an instance to fail, the deployment to that instance will not be considered to have failed at that point and will continue on to the BeforeInstall deployment lifecycle event.
            If set to false or not specified, then if the deployment causes the ApplicationStop deployment lifecycle event to fail to an instance, the deployment to that instance will stop, and the deployment to that instance will be considered to have failed.
            

    :type targetInstances: dict
    :param targetInstances: Information about the instances that will belong to the replacement environment in a blue/green deployment.
            tagFilters (list) --The tag filter key, type, and value used to identify Amazon EC2 instances in a replacement environment for a blue/green deployment.
            (dict) --Information about an EC2 tag filter.
            Key (string) --The tag filter key.
            Value (string) --The tag filter value.
            Type (string) --The tag filter type:
            KEY_ONLY: Key only.
            VALUE_ONLY: Value only.
            KEY_AND_VALUE: Key and value.
            
            autoScalingGroups (list) --The names of one or more Auto Scaling groups to identify a replacement environment for a blue/green deployment.
            (string) --
            

    :type autoRollbackConfiguration: dict
    :param autoRollbackConfiguration: Configuration information for an automatic rollback that is added when a deployment is created.
            enabled (boolean) --Indicates whether a defined automatic rollback configuration is currently enabled.
            events (list) --The event type or types that trigger a rollback.
            (string) --
            

    :type updateOutdatedInstancesOnly: boolean
    :param updateOutdatedInstancesOnly: Indicates whether to deploy to all instances or only to instances that are not running the latest application revision.

    :type fileExistsBehavior: string
    :param fileExistsBehavior: Information about how AWS CodeDeploy handles files that already exist in a deployment target location but weren't part of the previous successful deployment.
            The fileExistsBehavior parameter takes any of the following values:
            DISALLOW: The deployment fails. This is also the default behavior if no option is specified.
            OVERWRITE: The version of the file from the application revision currently being deployed replaces the version already on the instance.
            RETAIN: The version of the file already on the instance is kept and used as part of the new deployment.
            

    :rtype: dict
    :return: {
        'deploymentId': 'string'
    }
    
    
    """
    pass