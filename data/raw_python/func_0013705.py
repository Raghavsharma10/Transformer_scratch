def create_user_pool_client(UserPoolId=None, ClientName=None, GenerateSecret=None, RefreshTokenValidity=None, ReadAttributes=None, WriteAttributes=None, ExplicitAuthFlows=None, SupportedIdentityProviders=None, CallbackURLs=None, LogoutURLs=None, DefaultRedirectURI=None, AllowedOAuthFlows=None, AllowedOAuthScopes=None, AllowedOAuthFlowsUserPoolClient=None):
    """
    Creates the user pool client.
    See also: AWS API Documentation
    
    
    :example: response = client.create_user_pool_client(
        UserPoolId='string',
        ClientName='string',
        GenerateSecret=True|False,
        RefreshTokenValidity=123,
        ReadAttributes=[
            'string',
        ],
        WriteAttributes=[
            'string',
        ],
        ExplicitAuthFlows=[
            'ADMIN_NO_SRP_AUTH'|'CUSTOM_AUTH_FLOW_ONLY',
        ],
        SupportedIdentityProviders=[
            'string',
        ],
        CallbackURLs=[
            'string',
        ],
        LogoutURLs=[
            'string',
        ],
        DefaultRedirectURI='string',
        AllowedOAuthFlows=[
            'code'|'implicit'|'client_credentials',
        ],
        AllowedOAuthScopes=[
            'string',
        ],
        AllowedOAuthFlowsUserPoolClient=True|False
    )
    
    
    :type UserPoolId: string
    :param UserPoolId: [REQUIRED]
            The user pool ID for the user pool where you want to create a user pool client.
            

    :type ClientName: string
    :param ClientName: [REQUIRED]
            The client name for the user pool client you would like to create.
            

    :type GenerateSecret: boolean
    :param GenerateSecret: Boolean to specify whether you want to generate a secret for the user pool client being created.

    :type RefreshTokenValidity: integer
    :param RefreshTokenValidity: The time limit, in days, after which the refresh token is no longer valid and cannot be used.

    :type ReadAttributes: list
    :param ReadAttributes: The read attributes.
            (string) --
            

    :type WriteAttributes: list
    :param WriteAttributes: The write attributes.
            (string) --
            

    :type ExplicitAuthFlows: list
    :param ExplicitAuthFlows: The explicit authentication flows.
            (string) --
            

    :type SupportedIdentityProviders: list
    :param SupportedIdentityProviders: A list of provider names for the identity providers that are supported on this client.
            (string) --
            

    :type CallbackURLs: list
    :param CallbackURLs: A list of allowed callback URLs for the identity providers.
            (string) --
            

    :type LogoutURLs: list
    :param LogoutURLs: A list of allowed logout URLs for the identity providers.
            (string) --
            

    :type DefaultRedirectURI: string
    :param DefaultRedirectURI: The default redirect URI. Must be in the CallbackURLs list.

    :type AllowedOAuthFlows: list
    :param AllowedOAuthFlows: Set to code to initiate a code grant flow, which provides an authorization code as the response. This code can be exchanged for access tokens with the token endpoint.
            Set to token to specify that the client should get the access token (and, optionally, ID token, based on scopes) directly.
            (string) --
            

    :type AllowedOAuthScopes: list
    :param AllowedOAuthScopes: A list of allowed OAuth scopes. Currently supported values are 'phone' , 'email' , 'openid' , and 'Cognito' .
            (string) --
            

    :type AllowedOAuthFlowsUserPoolClient: boolean
    :param AllowedOAuthFlowsUserPoolClient: Set to True if the client is allowed to follow the OAuth protocol when interacting with Cognito user pools.

    :rtype: dict
    :return: {
        'UserPoolClient': {
            'UserPoolId': 'string',
            'ClientName': 'string',
            'ClientId': 'string',
            'ClientSecret': 'string',
            'LastModifiedDate': datetime(2015, 1, 1),
            'CreationDate': datetime(2015, 1, 1),
            'RefreshTokenValidity': 123,
            'ReadAttributes': [
                'string',
            ],
            'WriteAttributes': [
                'string',
            ],
            'ExplicitAuthFlows': [
                'ADMIN_NO_SRP_AUTH'|'CUSTOM_AUTH_FLOW_ONLY',
            ],
            'SupportedIdentityProviders': [
                'string',
            ],
            'CallbackURLs': [
                'string',
            ],
            'LogoutURLs': [
                'string',
            ],
            'DefaultRedirectURI': 'string',
            'AllowedOAuthFlows': [
                'code'|'implicit'|'client_credentials',
            ],
            'AllowedOAuthScopes': [
                'string',
            ],
            'AllowedOAuthFlowsUserPoolClient': True|False
        }
    }
    
    
    :returns: 
    (string) --
    
    """
    pass