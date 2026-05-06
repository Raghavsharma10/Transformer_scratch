def put_method(restApiId=None, resourceId=None, httpMethod=None, authorizationType=None, authorizerId=None, apiKeyRequired=None, operationName=None, requestParameters=None, requestModels=None, requestValidatorId=None):
    """
    Add a method to an existing  Resource resource.
    See also: AWS API Documentation
    
    
    :example: response = client.put_method(
        restApiId='string',
        resourceId='string',
        httpMethod='string',
        authorizationType='string',
        authorizerId='string',
        apiKeyRequired=True|False,
        operationName='string',
        requestParameters={
            'string': True|False
        },
        requestModels={
            'string': 'string'
        },
        requestValidatorId='string'
    )
    
    
    :type restApiId: string
    :param restApiId: [REQUIRED]
            The RestApi identifier for the new Method resource.
            

    :type resourceId: string
    :param resourceId: [REQUIRED]
            The Resource identifier for the new Method resource.
            

    :type httpMethod: string
    :param httpMethod: [REQUIRED]
            Specifies the method request's HTTP method type.
            

    :type authorizationType: string
    :param authorizationType: [REQUIRED]
            The method's authorization type. Valid values are NONE for open access, AWS_IAM for using AWS IAM permissions, CUSTOM for using a custom authorizer, or COGNITO_USER_POOLS for using a Cognito user pool.
            

    :type authorizerId: string
    :param authorizerId: Specifies the identifier of an Authorizer to use on this Method, if the type is CUSTOM.

    :type apiKeyRequired: boolean
    :param apiKeyRequired: Specifies whether the method required a valid ApiKey .

    :type operationName: string
    :param operationName: A human-friendly operation identifier for the method. For example, you can assign the operationName of ListPets for the GET /pets method in PetStore example.

    :type requestParameters: dict
    :param requestParameters: A key-value map defining required or optional method request parameters that can be accepted by Amazon API Gateway. A key defines a method request parameter name matching the pattern of method.request.{location}.{name} , where location is querystring , path , or header and name is a valid and unique parameter name. The value associated with the key is a Boolean flag indicating whether the parameter is required (true ) or optional (false ). The method request parameter names defined here are available in Integration to be mapped to integration request parameters or body-mapping templates.
            (string) --
            (boolean) --
            

    :type requestModels: dict
    :param requestModels: Specifies the Model resources used for the request's content type. Request models are represented as a key/value map, with a content type as the key and a Model name as the value.
            (string) --
            (string) --
            

    :type requestValidatorId: string
    :param requestValidatorId: The identifier of a RequestValidator for validating the method request.

    :rtype: dict
    :return: {
        'httpMethod': 'string',
        'authorizationType': 'string',
        'authorizerId': 'string',
        'apiKeyRequired': True|False,
        'requestValidatorId': 'string',
        'operationName': 'string',
        'requestParameters': {
            'string': True|False
        },
        'requestModels': {
            'string': 'string'
        },
        'methodResponses': {
            'string': {
                'statusCode': 'string',
                'responseParameters': {
                    'string': True|False
                },
                'responseModels': {
                    'string': 'string'
                }
            }
        },
        'methodIntegration': {
            'type': 'HTTP'|'AWS'|'MOCK'|'HTTP_PROXY'|'AWS_PROXY',
            'httpMethod': 'string',
            'uri': 'string',
            'credentials': 'string',
            'requestParameters': {
                'string': 'string'
            },
            'requestTemplates': {
                'string': 'string'
            },
            'passthroughBehavior': 'string',
            'contentHandling': 'CONVERT_TO_BINARY'|'CONVERT_TO_TEXT',
            'cacheNamespace': 'string',
            'cacheKeyParameters': [
                'string',
            ],
            'integrationResponses': {
                'string': {
                    'statusCode': 'string',
                    'selectionPattern': 'string',
                    'responseParameters': {
                        'string': 'string'
                    },
                    'responseTemplates': {
                        'string': 'string'
                    },
                    'contentHandling': 'CONVERT_TO_BINARY'|'CONVERT_TO_TEXT'
                }
            }
        }
    }
    
    
    :returns: 
    (string) --
    (boolean) --
    
    
    
    """
    pass