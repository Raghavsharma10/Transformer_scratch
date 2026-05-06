def modify_image_attribute(DryRun=None, ImageId=None, Attribute=None, OperationType=None, UserIds=None, UserGroups=None, ProductCodes=None, Value=None, LaunchPermission=None, Description=None):
    """
    Modifies the specified attribute of the specified AMI. You can specify only one attribute at a time.
    See also: AWS API Documentation
    
    
    :example: response = client.modify_image_attribute(
        DryRun=True|False,
        ImageId='string',
        Attribute='string',
        OperationType='add'|'remove',
        UserIds=[
            'string',
        ],
        UserGroups=[
            'string',
        ],
        ProductCodes=[
            'string',
        ],
        Value='string',
        LaunchPermission={
            'Add': [
                {
                    'UserId': 'string',
                    'Group': 'all'
                },
            ],
            'Remove': [
                {
                    'UserId': 'string',
                    'Group': 'all'
                },
            ]
        },
        Description={
            'Value': 'string'
        }
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type ImageId: string
    :param ImageId: [REQUIRED]
            The ID of the AMI.
            

    :type Attribute: string
    :param Attribute: The name of the attribute to modify.

    :type OperationType: string
    :param OperationType: The operation type.

    :type UserIds: list
    :param UserIds: One or more AWS account IDs. This is only valid when modifying the launchPermission attribute.
            (string) --
            

    :type UserGroups: list
    :param UserGroups: One or more user groups. This is only valid when modifying the launchPermission attribute.
            (string) --
            

    :type ProductCodes: list
    :param ProductCodes: One or more product codes. After you add a product code to an AMI, it can't be removed. This is only valid when modifying the productCodes attribute.
            (string) --
            

    :type Value: string
    :param Value: The value of the attribute being modified. This is only valid when modifying the description attribute.

    :type LaunchPermission: dict
    :param LaunchPermission: A launch permission modification.
            Add (list) --The AWS account ID to add to the list of launch permissions for the AMI.
            (dict) --Describes a launch permission.
            UserId (string) --The AWS account ID.
            Group (string) --The name of the group.
            
            Remove (list) --The AWS account ID to remove from the list of launch permissions for the AMI.
            (dict) --Describes a launch permission.
            UserId (string) --The AWS account ID.
            Group (string) --The name of the group.
            
            

    :type Description: dict
    :param Description: A description for the AMI.
            Value (string) --The attribute value. Note that the value is case-sensitive.
            

    """
    pass