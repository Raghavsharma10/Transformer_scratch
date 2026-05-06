def provision_product(AcceptLanguage=None, ProductId=None, ProvisioningArtifactId=None, PathId=None, ProvisionedProductName=None, ProvisioningParameters=None, Tags=None, NotificationArns=None, ProvisionToken=None):
    """
    Requests a Provision of a specified product. A ProvisionedProduct is a resourced instance for a product. For example, provisioning a CloudFormation-template-backed product results in launching a CloudFormation stack and all the underlying resources that come with it.
    You can check the status of this request using the  DescribeRecord operation.
    See also: AWS API Documentation
    
    
    :example: response = client.provision_product(
        AcceptLanguage='string',
        ProductId='string',
        ProvisioningArtifactId='string',
        PathId='string',
        ProvisionedProductName='string',
        ProvisioningParameters=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        Tags=[
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        NotificationArns=[
            'string',
        ],
        ProvisionToken='string'
    )
    
    
    :type AcceptLanguage: string
    :param AcceptLanguage: The language code to use for this operation. Supported language codes are as follows:
            'en' (English)
            'jp' (Japanese)
            'zh' (Chinese)
            If no code is specified, 'en' is used as the default.
            

    :type ProductId: string
    :param ProductId: [REQUIRED]
            The product identifier.
            

    :type ProvisioningArtifactId: string
    :param ProvisioningArtifactId: [REQUIRED]
            The provisioning artifact identifier for this product.
            

    :type PathId: string
    :param PathId: The identifier of the path for this product's provisioning. This value is optional if the product has a default path, and is required if there is more than one path for the specified product.

    :type ProvisionedProductName: string
    :param ProvisionedProductName: [REQUIRED]
            A user-friendly name to identify the ProvisionedProduct object. This value must be unique for the AWS account and cannot be updated after the product is provisioned.
            

    :type ProvisioningParameters: list
    :param ProvisioningParameters: Parameters specified by the administrator that are required for provisioning the product.
            (dict) --The arameter key/value pairs used to provision a product.
            Key (string) --The ProvisioningArtifactParameter.ParameterKey parameter from DescribeProvisioningParameters .
            Value (string) --The value to use for provisioning. Any constraints on this value can be found in ProvisioningArtifactParameter for Key .
            
            

    :type Tags: list
    :param Tags: A list of tags to use as provisioning options.
            (dict) --Key/value pairs to associate with this provisioning. These tags are entirely discretionary and are propagated to the resources created in the provisioning.
            Key (string) -- [REQUIRED]The ProvisioningArtifactParameter.TagKey parameter from DescribeProvisioningParameters .
            Value (string) -- [REQUIRED]The esired value for this key.
            
            

    :type NotificationArns: list
    :param NotificationArns: Passed to CloudFormation. The SNS topic ARNs to which to publish stack-related events.
            (string) --
            

    :type ProvisionToken: string
    :param ProvisionToken: [REQUIRED]
            An idempotency token that uniquely identifies the provisioning request.
            This field is autopopulated if not provided.
            

    :rtype: dict
    :return: {
        'RecordDetail': {
            'RecordId': 'string',
            'ProvisionedProductName': 'string',
            'Status': 'IN_PROGRESS'|'SUCCEEDED'|'ERROR',
            'CreatedTime': datetime(2015, 1, 1),
            'UpdatedTime': datetime(2015, 1, 1),
            'ProvisionedProductType': 'string',
            'RecordType': 'string',
            'ProvisionedProductId': 'string',
            'ProductId': 'string',
            'ProvisioningArtifactId': 'string',
            'PathId': 'string',
            'RecordErrors': [
                {
                    'Code': 'string',
                    'Description': 'string'
                },
            ],
            'RecordTags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        }
    }
    
    
    """
    pass