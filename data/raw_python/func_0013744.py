def update_product(AcceptLanguage=None, Id=None, Name=None, Owner=None, Description=None, Distributor=None, SupportDescription=None, SupportEmail=None, SupportUrl=None, AddTags=None, RemoveTags=None):
    """
    Updates an existing product.
    See also: AWS API Documentation
    
    
    :example: response = client.update_product(
        AcceptLanguage='string',
        Id='string',
        Name='string',
        Owner='string',
        Description='string',
        Distributor='string',
        SupportDescription='string',
        SupportEmail='string',
        SupportUrl='string',
        AddTags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        RemoveTags=[
            'string',
        ]
    )
    
    
    :type AcceptLanguage: string
    :param AcceptLanguage: The language code to use for this operation. Supported language codes are as follows:
            'en' (English)
            'jp' (Japanese)
            'zh' (Chinese)
            If no code is specified, 'en' is used as the default.
            

    :type Id: string
    :param Id: [REQUIRED]
            The identifier of the product for the update request.
            

    :type Name: string
    :param Name: The updated product name.

    :type Owner: string
    :param Owner: The updated owner of the product.

    :type Description: string
    :param Description: The updated text description of the product.

    :type Distributor: string
    :param Distributor: The updated distributor of the product.

    :type SupportDescription: string
    :param SupportDescription: The updated support description for the product.

    :type SupportEmail: string
    :param SupportEmail: The updated support email for the product.

    :type SupportUrl: string
    :param SupportUrl: The updated support URL for the product.

    :type AddTags: list
    :param AddTags: Tags to add to the existing list of tags associated with the product.
            (dict) --Key/value pairs to associate with this provisioning. These tags are entirely discretionary and are propagated to the resources created in the provisioning.
            Key (string) -- [REQUIRED]The ProvisioningArtifactParameter.TagKey parameter from DescribeProvisioningParameters .
            Value (string) -- [REQUIRED]The esired value for this key.
            
            

    :type RemoveTags: list
    :param RemoveTags: Tags to remove from the existing list of tags associated with the product.
            (string) --
            

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
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
            },
        ]
    }
    
    
    """
    pass