def create_app(StackId=None, Shortname=None, Name=None, Description=None, DataSources=None, Type=None, AppSource=None, Domains=None, EnableSsl=None, SslConfiguration=None, Attributes=None, Environment=None):
    """
    Creates an app for a specified stack. For more information, see Creating Apps .
    See also: AWS API Documentation
    
    
    :example: response = client.create_app(
        StackId='string',
        Shortname='string',
        Name='string',
        Description='string',
        DataSources=[
            {
                'Type': 'string',
                'Arn': 'string',
                'DatabaseName': 'string'
            },
        ],
        Type='aws-flow-ruby'|'java'|'rails'|'php'|'nodejs'|'static'|'other',
        AppSource={
            'Type': 'git'|'svn'|'archive'|'s3',
            'Url': 'string',
            'Username': 'string',
            'Password': 'string',
            'SshKey': 'string',
            'Revision': 'string'
        },
        Domains=[
            'string',
        ],
        EnableSsl=True|False,
        SslConfiguration={
            'Certificate': 'string',
            'PrivateKey': 'string',
            'Chain': 'string'
        },
        Attributes={
            'string': 'string'
        },
        Environment=[
            {
                'Key': 'string',
                'Value': 'string',
                'Secure': True|False
            },
        ]
    )
    
    
    :type StackId: string
    :param StackId: [REQUIRED]
            The stack ID.
            

    :type Shortname: string
    :param Shortname: The app's short name.

    :type Name: string
    :param Name: [REQUIRED]
            The app name.
            

    :type Description: string
    :param Description: A description of the app.

    :type DataSources: list
    :param DataSources: The app's data source.
            (dict) --Describes an app's data source.
            Type (string) --The data source's type, AutoSelectOpsworksMysqlInstance , OpsworksMysqlInstance , or RdsDbInstance .
            Arn (string) --The data source's ARN.
            DatabaseName (string) --The database name.
            
            

    :type Type: string
    :param Type: [REQUIRED]
            The app type. Each supported type is associated with a particular layer. For example, PHP applications are associated with a PHP layer. AWS OpsWorks Stacks deploys an application to those instances that are members of the corresponding layer. If your app isn't one of the standard types, or you prefer to implement your own Deploy recipes, specify other .
            

    :type AppSource: dict
    :param AppSource: A Source object that specifies the app repository.
            Type (string) --The repository type.
            Url (string) --The source URL.
            Username (string) --This parameter depends on the repository type.
            For Amazon S3 bundles, set Username to the appropriate IAM access key ID.
            For HTTP bundles, Git repositories, and Subversion repositories, set Username to the user name.
            Password (string) --When included in a request, the parameter depends on the repository type.
            For Amazon S3 bundles, set Password to the appropriate IAM secret access key.
            For HTTP bundles and Subversion repositories, set Password to the password.
            For more information on how to safely handle IAM credentials, see http://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html .
            In responses, AWS OpsWorks Stacks returns *****FILTERED***** instead of the actual value.
            SshKey (string) --In requests, the repository's SSH key.
            In responses, AWS OpsWorks Stacks returns *****FILTERED***** instead of the actual value.
            Revision (string) --The application's version. AWS OpsWorks Stacks enables you to easily deploy new versions of an application. One of the simplest approaches is to have branches or revisions in your repository that represent different versions that can potentially be deployed.
            

    :type Domains: list
    :param Domains: The app virtual host settings, with multiple domains separated by commas. For example: 'www.example.com, example.com'
            (string) --
            

    :type EnableSsl: boolean
    :param EnableSsl: Whether to enable SSL for the app.

    :type SslConfiguration: dict
    :param SslConfiguration: An SslConfiguration object with the SSL configuration.
            Certificate (string) -- [REQUIRED]The contents of the certificate's domain.crt file.
            PrivateKey (string) -- [REQUIRED]The private key; the contents of the certificate's domain.kex file.
            Chain (string) --Optional. Can be used to specify an intermediate certificate authority key or client authentication.
            

    :type Attributes: dict
    :param Attributes: One or more user-defined key/value pairs to be added to the stack attributes.
            (string) --
            (string) --
            

    :type Environment: list
    :param Environment: An array of EnvironmentVariable objects that specify environment variables to be associated with the app. After you deploy the app, these variables are defined on the associated app server instance. For more information, see Environment Variables .
            There is no specific limit on the number of environment variables. However, the size of the associated data structure - which includes the variables' names, values, and protected flag values - cannot exceed 10 KB (10240 Bytes). This limit should accommodate most if not all use cases. Exceeding it will cause an exception with the message, 'Environment: is too large (maximum is 10KB).'
            Note
            This parameter is supported only by Chef 11.10 stacks. If you have specified one or more environment variables, you cannot modify the stack's Chef version.
            (dict) --Represents an app's environment variable.
            Key (string) -- [REQUIRED](Required) The environment variable's name, which can consist of up to 64 characters and must be specified. The name can contain upper- and lowercase letters, numbers, and underscores (_), but it must start with a letter or underscore.
            Value (string) -- [REQUIRED](Optional) The environment variable's value, which can be left empty. If you specify a value, it can contain up to 256 characters, which must all be printable.
            Secure (boolean) --(Optional) Whether the variable's value will be returned by the DescribeApps action. To conceal an environment variable's value, set Secure to true . DescribeApps then returns *****FILTERED***** instead of the actual value. The default value for Secure is false .
            
            

    :rtype: dict
    :return: {
        'AppId': 'string'
    }
    
    
    """
    pass