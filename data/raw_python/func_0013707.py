def update_user_pool_client(UserPoolId=None, ClientId=None, ClientName=None, RefreshTokenValidity=None, ReadAttributes=None, WriteAttributes=None, ExplicitAuthFlows=None, SupportedIdentityProviders=None, CallbackURLs=None, LogoutURLs=None, DefaultRedirectURI=None, AllowedOAuthFlows=None, AllowedOAuthScopes=None, AllowedOAuthFlowsUserPoolClient=None):
    """
    Allows the developer to update the specified user pool client and password policy.
    See also: AWS API Documentation
    
    
    :example: response = client.update_user_pool_client(
        UserPoolId='string',
        ClientId='string',
        ClientName='string',
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
            The user pool ID for the user pool where you want to update the user pool client.
            

    :type ClientId: string
    :param ClientId: [REQUIRED]
            The ID of the client associated with the user pool.
            

    :type ClientName: string
    :param ClientName: The client name from the update user pool client request.

    :type RefreshTokenValidity: integer
    :param RefreshTokenValidity: The time limit, in days, after which the refresh token is no longer valid and cannot be used.

    :type ReadAttributes: list
    :param ReadAttributes: The read-only attributes of the user pool.
            (string) --
            

    :type WriteAttributes: list
    :param WriteAttributes: The writeable attributes of the user pool.
            (string) --
            

    :type ExplicitAuthFlows: list
    :param ExplicitAuthFlows: Explicit authentication flows.
            (string) --
            

    :type SupportedIdentityProviders: list
    :param SupportedIdentityProviders: A list of provider names for the identity providers that are supported on this client.
            (string) --
            

    :type CallbackURLs: list
    :param CallbackURLs: A list of allowed callback URLs for the identity providers.
            (string) --
            

    :type LogoutURLs: list
    :param LogoutURLs: A list ofallowed logout URLs for the identity providers.
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
    :param AllowedOAuthFlowsUserPoolClient: Set to TRUE if the client is allowed to follow the OAuth protocol when interacting with Cognito user pools.

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