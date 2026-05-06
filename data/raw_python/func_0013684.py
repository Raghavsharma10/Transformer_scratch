def create_project(name=None, description=None, source=None, artifacts=None, environment=None, serviceRole=None, timeoutInMinutes=None, encryptionKey=None, tags=None):
    """
    Creates a build project.
    See also: AWS API Documentation
    
    
    :example: response = client.create_project(
        name='string',
        description='string',
        source={
            'type': 'CODECOMMIT'|'CODEPIPELINE'|'GITHUB'|'S3',
            'location': 'string',
            'buildspec': 'string',
            'auth': {
                'type': 'OAUTH',
                'resource': 'string'
            }
        },
        artifacts={
            'type': 'CODEPIPELINE'|'S3'|'NO_ARTIFACTS',
            'location': 'string',
            'path': 'string',
            'namespaceType': 'NONE'|'BUILD_ID',
            'name': 'string',
            'packaging': 'NONE'|'ZIP'
        },
        environment={
            'type': 'LINUX_CONTAINER',
            'image': 'string',
            'computeType': 'BUILD_GENERAL1_SMALL'|'BUILD_GENERAL1_MEDIUM'|'BUILD_GENERAL1_LARGE',
            'environmentVariables': [
                {
                    'name': 'string',
                    'value': 'string'
                },
            ]
        },
        serviceRole='string',
        timeoutInMinutes=123,
        encryptionKey='string',
        tags=[
            {
                'key': 'string',
                'value': 'string'
            },
        ]
    )
    
    
    :type name: string
    :param name: [REQUIRED]
            The name of the build project.
            

    :type description: string
    :param description: A description that makes the build project easy to identify.

    :type source: dict
    :param source: [REQUIRED]
            Information about the build input source code for the build project.
            type (string) -- [REQUIRED]The type of repository that contains the source code to be built. Valid values include:
            CODECOMMIT : The source code is in an AWS CodeCommit repository.
            CODEPIPELINE : The source code settings are specified in the source action of a pipeline in AWS CodePipeline.
            GITHUB : The source code is in a GitHub repository.
            S3 : The source code is in an Amazon Simple Storage Service (Amazon S3) input bucket.
            location (string) --Information about the location of the source code to be built. Valid values include:
            For source code settings that are specified in the source action of a pipeline in AWS CodePipeline, location should not be specified. If it is specified, AWS CodePipeline will ignore it. This is because AWS CodePipeline uses the settings in a pipeline's source action instead of this value.
            For source code in an AWS CodeCommit repository, the HTTPS clone URL to the repository that contains the source code and the build spec (for example, ``https://git-codecommit.*region-ID* .amazonaws.com/v1/repos/repo-name `` ).
            For source code in an Amazon Simple Storage Service (Amazon S3) input bucket, the path to the ZIP file that contains the source code (for example, `` bucket-name /path /to /object-name .zip`` )
            For source code in a GitHub repository, the HTTPS clone URL to the repository that contains the source and the build spec. Also, you must connect your AWS account to your GitHub account. To do this, use the AWS CodeBuild console to begin creating a build project, and follow the on-screen instructions to complete the connection. (After you have connected to your GitHub account, you do not need to finish creating the build project, and you may then leave the AWS CodeBuild console.) To instruct AWS CodeBuild to then use this connection, in the source object, set the auth object's type value to OAUTH .
            buildspec (string) --The build spec declaration to use for the builds in this build project.
            If this value is not specified, a build spec must be included along with the source code to be built.
            auth (dict) --Information about the authorization settings for AWS CodeBuild to access the source code to be built.
            This information is for the AWS CodeBuild console's use only. Your code should not get or set this information directly (unless the build project's source type value is GITHUB ).
            type (string) -- [REQUIRED]The authorization type to use. The only valid value is OAUTH , which represents the OAuth authorization type.
            resource (string) --The resource value that applies to the specified authorization type.
            
            

    :type artifacts: dict
    :param artifacts: [REQUIRED]
            Information about the build output artifacts for the build project.
            type (string) -- [REQUIRED]The type of build output artifact. Valid values include:
            CODEPIPELINE : The build project will have build output generated through AWS CodePipeline.
            NO_ARTIFACTS : The build project will not produce any build output.
            S3 : The build project will store build output in Amazon Simple Storage Service (Amazon S3).
            location (string) --Information about the build output artifact location, as follows:
            If type is set to CODEPIPELINE , then AWS CodePipeline will ignore this value if specified. This is because AWS CodePipeline manages its build output locations instead of AWS CodeBuild.
            If type is set to NO_ARTIFACTS , then this value will be ignored if specified, because no build output will be produced.
            If type is set to S3 , this is the name of the output bucket.
            path (string) --Along with namespaceType and name , the pattern that AWS CodeBuild will use to name and store the output artifact, as follows:
            If type is set to CODEPIPELINE , then AWS CodePipeline will ignore this value if specified. This is because AWS CodePipeline manages its build output names instead of AWS CodeBuild.
            If type is set to NO_ARTIFACTS , then this value will be ignored if specified, because no build output will be produced.
            If type is set to S3 , this is the path to the output artifact. If path is not specified, then path will not be used.
            For example, if path is set to MyArtifacts , namespaceType is set to NONE , and name is set to MyArtifact.zip , then the output artifact would be stored in the output bucket at MyArtifacts/MyArtifact.zip .
            namespaceType (string) --Along with path and name , the pattern that AWS CodeBuild will use to determine the name and location to store the output artifact, as follows:
            If type is set to CODEPIPELINE , then AWS CodePipeline will ignore this value if specified. This is because AWS CodePipeline manages its build output names instead of AWS CodeBuild.
            If type is set to NO_ARTIFACTS , then this value will be ignored if specified, because no build output will be produced.
            If type is set to S3 , then valid values include:
            BUILD_ID : Include the build ID in the location of the build output artifact.
            NONE : Do not include the build ID. This is the default if namespaceType is not specified.
            For example, if path is set to MyArtifacts , namespaceType is set to BUILD_ID , and name is set to MyArtifact.zip , then the output artifact would be stored in MyArtifacts/*build-ID* /MyArtifact.zip .
            name (string) --Along with path and namespaceType , the pattern that AWS CodeBuild will use to name and store the output artifact, as follows:
            If type is set to CODEPIPELINE , then AWS CodePipeline will ignore this value if specified. This is because AWS CodePipeline manages its build output names instead of AWS CodeBuild.
            If type is set to NO_ARTIFACTS , then this value will be ignored if specified, because no build output will be produced.
            If type is set to S3 , this is the name of the output artifact object.
            For example, if path is set to MyArtifacts , namespaceType is set to BUILD_ID , and name is set to MyArtifact.zip , then the output artifact would be stored in MyArtifacts/*build-ID* /MyArtifact.zip .
            packaging (string) --The type of build output artifact to create, as follows:
            If type is set to CODEPIPELINE , then AWS CodePipeline will ignore this value if specified. This is because AWS CodePipeline manages its build output artifacts instead of AWS CodeBuild.
            If type is set to NO_ARTIFACTS , then this value will be ignored if specified, because no build output will be produced.
            If type is set to S3 , valid values include:
            NONE : AWS CodeBuild will create in the output bucket a folder containing the build output. This is the default if packaging is not specified.
            ZIP : AWS CodeBuild will create in the output bucket a ZIP file containing the build output.
            
            

    :type environment: dict
    :param environment: [REQUIRED]
            Information about the build environment for the build project.
            type (string) -- [REQUIRED]The type of build environment to use for related builds.
            image (string) -- [REQUIRED]The ID of the Docker image to use for this build project.
            computeType (string) -- [REQUIRED]Information about the compute resources the build project will use. Available values include:
            BUILD_GENERAL1_SMALL : Use up to 3 GB memory and 2 vCPUs for builds.
            BUILD_GENERAL1_MEDIUM : Use up to 7 GB memory and 4 vCPUs for builds.
            BUILD_GENERAL1_LARGE : Use up to 15 GB memory and 8 vCPUs for builds.
            environmentVariables (list) --A set of environment variables to make available to builds for this build project.
            (dict) --Information about an environment variable for a build project or a build.
            name (string) -- [REQUIRED]The name or key of the environment variable.
            value (string) -- [REQUIRED]The value of the environment variable.
            
            

    :type serviceRole: string
    :param serviceRole: The ARN of the AWS Identity and Access Management (IAM) role that enables AWS CodeBuild to interact with dependent AWS services on behalf of the AWS account.

    :type timeoutInMinutes: integer
    :param timeoutInMinutes: How long, in minutes, from 5 to 480 (8 hours), for AWS CodeBuild to wait until timing out any build that has not been marked as completed. The default is 60 minutes.

    :type encryptionKey: string
    :param encryptionKey: The AWS Key Management Service (AWS KMS) customer master key (CMK) to be used for encrypting the build output artifacts.
            You can specify either the CMK's Amazon Resource Name (ARN) or, if available, the CMK's alias (using the format ``alias/alias-name `` ).
            

    :type tags: list
    :param tags: A set of tags for this build project.
            These tags are available for use by AWS services that support AWS CodeBuild build project tags.
            (dict) --A tag, consisting of a key and a value.
            This tag is available for use by AWS services that support tags in AWS CodeBuild.
            key (string) --The tag's key.
            value (string) --The tag's value.
            
            

    :rtype: dict
    :return: {
        'project': {
            'name': 'string',
            'arn': 'string',
            'description': 'string',
            'source': {
                'type': 'CODECOMMIT'|'CODEPIPELINE'|'GITHUB'|'S3',
                'location': 'string',
                'buildspec': 'string',
                'auth': {
                    'type': 'OAUTH',
                    'resource': 'string'
                }
            },
            'artifacts': {
                'type': 'CODEPIPELINE'|'S3'|'NO_ARTIFACTS',
                'location': 'string',
                'path': 'string',
                'namespaceType': 'NONE'|'BUILD_ID',
                'name': 'string',
                'packaging': 'NONE'|'ZIP'
            },
            'environment': {
                'type': 'LINUX_CONTAINER',
                'image': 'string',
                'computeType': 'BUILD_GENERAL1_SMALL'|'BUILD_GENERAL1_MEDIUM'|'BUILD_GENERAL1_LARGE',
                'environmentVariables': [
                    {
                        'name': 'string',
                        'value': 'string'
                    },
                ]
            },
            'serviceRole': 'string',
            'timeoutInMinutes': 123,
            'encryptionKey': 'string',
            'tags': [
                {
                    'key': 'string',
                    'value': 'string'
                },
            ],
            'created': datetime(2015, 1, 1),
            'lastModified': datetime(2015, 1, 1)
        }
    }
    
    
    :returns: 
    CODECOMMIT : The source code is in an AWS CodeCommit repository.
    CODEPIPELINE : The source code settings are specified in the source action of a pipeline in AWS CodePipeline.
    GITHUB : The source code is in a GitHub repository.
    S3 : The source code is in an Amazon Simple Storage Service (Amazon S3) input bucket.
    
    """
    pass