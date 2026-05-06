def create_product(AcceptLanguage=None, Name=None, Owner=None, Description=None, Distributor=None, SupportDescription=None, SupportEmail=None, SupportUrl=None, ProductType=None, Tags=None, ProvisioningArtifactParameters=None, IdempotencyToken=None):
    """
    Creates a new product.
    See also: AWS API Documentation
    
    
    :example: response = client.create_product(
        AcceptLanguage='string',
        Name='string',
        Owner='string',
        Description='string',
        Distributor='string',
        SupportDescription='string',
        SupportEmail='string',
        SupportUrl='string',
        ProductType='CLOUD_FORMATION_TEMPLATE',
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        ProvisioningArtifactParameters={
            'Name': 'string',
            'Description': 'string',
            'Info': {
                'string': 'string'
            },
            'Type': 'CLOUD_FORMATION_TEMPLATE'
        },
        IdempotencyToken='string'
    )
    
    
    :type AcceptLanguage: string
    :param AcceptLanguage: The language code to use for this operation. Supported language codes are as follows:
            'en' (English)
            'jp' (Japanese)
            'zh' (Chinese)
            If no code is specified, 'en' is used as the default.
            

    :type Name: string
    :param Name: [REQUIRED]
            The name of the product.
            

    :type Owner: string
    :param Owner: [REQUIRED]
            The owner of the product.
            

    :type Description: string
    :param Description: The text description of the product.

    :type Distributor: string
    :param Distributor: The distributor of the product.

    :type SupportDescription: string
    :param SupportDescription: Support information about the product.

    :type SupportEmail: string
    :param SupportEmail: Contact email for product support.

    :type SupportUrl: string
    :param SupportUrl: Contact URL for product support.

    :type ProductType: string
    :param ProductType: [REQUIRED]
            The type of the product to create.
            

    :type Tags: list
    :param Tags: Tags to associate with the new product.
            (dict) --Key/value pairs to associate with this provisioning. These tags are entirely discretionary and are propagated to the resources created in the provisioning.
            Key (string) -- [REQUIRED]The ProvisioningArtifactParameter.TagKey parameter from DescribeProvisioningParameters .
            Value (string) -- [REQUIRED]The esired value for this key.
            
            

    :type ProvisioningArtifactParameters: dict
    :param ProvisioningArtifactParameters: [REQUIRED]
            Parameters for the provisioning artifact.
            Name (string) --The name assigned to the provisioning artifact properties.
            Description (string) --The text description of the provisioning artifact properties.
            Info (dict) -- [REQUIRED]Additional information about the provisioning artifact properties.
            (string) --
            (string) --
            
            Type (string) --The type of the provisioning artifact properties.
            

    :type IdempotencyToken: string
    :param IdempotencyToken: [REQUIRED]
            A token to disambiguate duplicate requests. You can create multiple resources using the same input in multiple requests, provided that you also specify a different idempotency token for each request.
            This field is autopopulated if not provided.
            

    :rtype: dict
    :return: {
        'ProductViewDetail': {
            'ProductViewSummary': {
                'Id': 'string',
                'ProductId': 'string',
                'Name': 'string',
                'Owner': 'string',
                'ShortDescription': 'string',
                'Type': 'CLOUD_FORMATION_TEMPLATE',
                'Distributor': 'string',
                'HasDefaultPath': True|False,
                'SupportEmail': 'string',
                'SupportDescription': 'string',
                'SupportUrl': 'string'
            },
            'Status': 'AVAILABLE'|'CREATING'|'FAILED',
            'ProductARN': 'string',
            'CreatedTime': datetime(2015, 1, 1)
        },
        'ProvisioningArtifactDetail': {
            'Id': 'string',
            'Name': 'string',
            'Description': 'string',
            'Type': 'CLOUD_FORMATION_TEMPLATE',
            'CreatedTime': datetime(2015, 1, 1)
        },
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
            },
        ]
    }
    
    
    """
    pass