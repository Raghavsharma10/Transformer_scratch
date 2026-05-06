def create_authorizer(restApiId=None, name=None, type=None, providerARNs=None, authType=None, authorizerUri=None, authorizerCredentials=None, identitySource=None, identityValidationExpression=None, authorizerResultTtlInSeconds=None):
    """
    Adds a new  Authorizer resource to an existing  RestApi resource.
    See also: AWS API Documentation
    
    
    :example: response = client.create_authorizer(
        restApiId='string',
        name='string',
        type='TOKEN'|'COGNITO_USER_POOLS',
        providerARNs=[
            'string',
        ],
        authType='string',
        authorizerUri='string',
        authorizerCredentials='string',
        identitySource='string',
        identityValidationExpression='string',
        authorizerResultTtlInSeconds=123
    )
    
    
    :type restApiId: string
    :param restApiId: [REQUIRED]
            The RestApi identifier under which the Authorizer will be created.
            

    :type name: string
    :param name: [REQUIRED]
            [Required] The name of the authorizer.
            

    :type type: string
    :param type: [REQUIRED]
            [Required] The type of the authorizer.
            

    :type providerARNs: list
    :param providerARNs: A list of the Cognito Your User Pool authorizer's provider ARNs.
            (string) --
            

    :type authType: string
    :param authType: Optional customer-defined field, used in Swagger imports/exports. Has no functional impact.

    :type authorizerUri: string
    :param authorizerUri: [Required] Specifies the authorizer's Uniform Resource Identifier (URI).

    :type authorizerCredentials: string
    :param authorizerCredentials: Specifies the credentials required for the authorizer, if any.

    :type identitySource: string
    :param identitySource: [REQUIRED]
            [Required] The source of the identity in an incoming request.
            

    :type identityValidationExpression: string
    :param identityValidationExpression: A validation expression for the incoming identity.

    :type authorizerResultTtlInSeconds: integer
    :param authorizerResultTtlInSeconds: The TTL of cached authorizer results.

    :rtype: dict
    :return: {
        'id': 'string',
        'name': 'string',
        'type': 'TOKEN'|'COGNITO_USER_POOLS',
        'providerARNs': [
            'string',
        ],
        'authType': 'string',
        'authorizerUri': 'string',
        'authorizerCredentials': 'string',
        'identitySource': 'string',
        'identityValidationExpression': 'string',
        'authorizerResultTtlInSeconds': 123
    }
    
    
    :returns: 
    (string) --
    
    """
    pass