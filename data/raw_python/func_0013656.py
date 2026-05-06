def modify_endpoint(EndpointArn=None, EndpointIdentifier=None, EndpointType=None, EngineName=None, Username=None, Password=None, ServerName=None, Port=None, DatabaseName=None, ExtraConnectionAttributes=None, CertificateArn=None, SslMode=None, DynamoDbSettings=None, S3Settings=None, MongoDbSettings=None):
    """
    Modifies the specified endpoint.
    See also: AWS API Documentation
    
    
    :example: response = client.modify_endpoint(
        EndpointArn='string',
        EndpointIdentifier='string',
        EndpointType='source'|'target',
        EngineName='string',
        Username='string',
        Password='string',
        ServerName='string',
        Port=123,
        DatabaseName='string',
        ExtraConnectionAttributes='string',
        CertificateArn='string',
        SslMode='none'|'require'|'verify-ca'|'verify-full',
        DynamoDbSettings={
            'ServiceAccessRoleArn': 'string'
        },
        S3Settings={
            'ServiceAccessRoleArn': 'string',
            'ExternalTableDefinition': 'string',
            'CsvRowDelimiter': 'string',
            'CsvDelimiter': 'string',
            'BucketFolder': 'string',
            'BucketName': 'string',
            'CompressionType': 'none'|'gzip'
        },
        MongoDbSettings={
            'Username': 'string',
            'Password': 'string',
            'ServerName': 'string',
            'Port': 123,
            'DatabaseName': 'string',
            'AuthType': 'no'|'password',
            'AuthMechanism': 'default'|'mongodb_cr'|'scram_sha_1',
            'NestingLevel': 'none'|'one',
            'ExtractDocId': 'string',
            'DocsToInvestigate': 'string',
            'AuthSource': 'string'
        }
    )
    
    
    :type EndpointArn: string
    :param EndpointArn: [REQUIRED]
            The Amazon Resource Name (ARN) string that uniquely identifies the endpoint.
            

    :type EndpointIdentifier: string
    :param EndpointIdentifier: The database endpoint identifier. Identifiers must begin with a letter; must contain only ASCII letters, digits, and hyphens; and must not end with a hyphen or contain two consecutive hyphens.

    :type EndpointType: string
    :param EndpointType: The type of endpoint.

    :type EngineName: string
    :param EngineName: The type of engine for the endpoint. Valid values, depending on the EndPointType, include MYSQL, ORACLE, POSTGRES, MARIADB, AURORA, REDSHIFT, S3, DYNAMODB, MONGODB, SYBASE, and SQLSERVER.

    :type Username: string
    :param Username: The user name to be used to login to the endpoint database.

    :type Password: string
    :param Password: The password to be used to login to the endpoint database.

    :type ServerName: string
    :param ServerName: The name of the server where the endpoint database resides.

    :type Port: integer
    :param Port: The port used by the endpoint database.

    :type DatabaseName: string
    :param DatabaseName: The name of the endpoint database.

    :type ExtraConnectionAttributes: string
    :param ExtraConnectionAttributes: Additional attributes associated with the connection.

    :type CertificateArn: string
    :param CertificateArn: The Amazon Resource Name (ARN) of the certificate used for SSL connection.

    :type SslMode: string
    :param SslMode: The SSL mode to be used.
            SSL mode can be one of four values: none, require, verify-ca, verify-full.
            The default value is none.
            

    :type DynamoDbSettings: dict
    :param DynamoDbSettings: Settings in JSON format for the target Amazon DynamoDB endpoint. For more information about the available settings, see the Using Object Mapping to Migrate Data to DynamoDB section at Using an Amazon DynamoDB Database as a Target for AWS Database Migration Service .
            ServiceAccessRoleArn (string) -- [REQUIRED]The Amazon Resource Name (ARN) used by the service access IAM role.
            

    :type S3Settings: dict
    :param S3Settings: Settings in JSON format for the target S3 endpoint. For more information about the available settings, see the Extra Connection Attributes section at Using Amazon S3 as a Target for AWS Database Migration Service .
            ServiceAccessRoleArn (string) --The Amazon Resource Name (ARN) used by the service access IAM role.
            ExternalTableDefinition (string) --
            CsvRowDelimiter (string) --The delimiter used to separate rows in the source files. The default is a carriage return (n).
            CsvDelimiter (string) --The delimiter used to separate columns in the source files. The default is a comma.
            BucketFolder (string) --An optional parameter to set a folder name in the S3 bucket. If provided, tables are created in the path bucketFolder/schema_name/table_name/. If this parameter is not specified, then the path used is schema_name/table_name/.
            BucketName (string) --The name of the S3 bucket.
            CompressionType (string) --An optional parameter to use GZIP to compress the target files. Set to GZIP to compress the target files. Set to NONE (the default) or do not use to leave the files uncompressed.
            

    :type MongoDbSettings: dict
    :param MongoDbSettings: Settings in JSON format for the source MongoDB endpoint. For more information about the available settings, see the Configuration Properties When Using MongoDB as a Source for AWS Database Migration Service section at Using Amazon S3 as a Target for AWS Database Migration Service .
            Username (string) --The user name you use to access the MongoDB source endpoint.
            Password (string) --The password for the user account you use to access the MongoDB source endpoint.
            ServerName (string) --The name of the server on the MongoDB source endpoint.
            Port (integer) --The port value for the MongoDB source endpoint.
            DatabaseName (string) --The database name on the MongoDB source endpoint.
            AuthType (string) --The authentication type you use to access the MongoDB source endpoint.
            Valid values: NO, PASSWORD
            When NO is selected, user name and password parameters are not used and can be empty.
            AuthMechanism (string) --The authentication mechanism you use to access the MongoDB source endpoint.
            Valid values: DEFAULT, MONGODB_CR, SCRAM_SHA_1
            DEFAULT   For MongoDB version 2.x, use MONGODB_CR. For MongoDB version 3.x, use SCRAM_SHA_1. This attribute is not used when authType=No.
            NestingLevel (string) --Specifies either document or table mode.
            Valid values: NONE, ONE
            Default value is NONE. Specify NONE to use document mode. Specify ONE to use table mode.
            ExtractDocId (string) --Specifies the document ID. Use this attribute when NestingLevel is set to NONE.
            Default value is false.
            DocsToInvestigate (string) --Indicates the number of documents to preview to determine the document organization. Use this attribute when NestingLevel is set to ONE.
            Must be a positive value greater than 0. Default value is 1000.
            AuthSource (string) --The MongoDB database name. This attribute is not used when authType=NO .
            The default is admin.
            

    :rtype: dict
    :return: {
        'Endpoint': {
            'EndpointIdentifier': 'string',
            'EndpointType': 'source'|'target',
            'EngineName': 'string',
            'Username': 'string',
            'ServerName': 'string',
            'Port': 123,
            'DatabaseName': 'string',
            'ExtraConnectionAttributes': 'string',
            'Status': 'string',
            'KmsKeyId': 'string',
            'EndpointArn': 'string',
            'CertificateArn': 'string',
            'SslMode': 'none'|'require'|'verify-ca'|'verify-full',
            'ExternalId': 'string',
            'DynamoDbSettings': {
                'ServiceAccessRoleArn': 'string'
            },
            'S3Settings': {
                'ServiceAccessRoleArn': 'string',
                'ExternalTableDefinition': 'string',
                'CsvRowDelimiter': 'string',
                'CsvDelimiter': 'string',
                'BucketFolder': 'string',
                'BucketName': 'string',
                'CompressionType': 'none'|'gzip'
            },
            'MongoDbSettings': {
                'Username': 'string',
                'Password': 'string',
                'ServerName': 'string',
                'Port': 123,
                'DatabaseName': 'string',
                'AuthType': 'no'|'password',
                'AuthMechanism': 'default'|'mongodb_cr'|'scram_sha_1',
                'NestingLevel': 'none'|'one',
                'ExtractDocId': 'string',
                'DocsToInvestigate': 'string',
                'AuthSource': 'string'
            }
        }
    }
    
    
    """
    pass