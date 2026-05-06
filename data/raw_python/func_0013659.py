def update_fleet(ImageName=None, Name=None, InstanceType=None, ComputeCapacity=None, VpcConfig=None, MaxUserDurationInSeconds=None, DisconnectTimeoutInSeconds=None, DeleteVpcConfig=None, Description=None, DisplayName=None, EnableDefaultInternetAccess=None):
    """
    Updates an existing fleet. All the attributes except the fleet name can be updated in the STOPPED state. When a fleet is in the RUNNING state, only DisplayName and ComputeCapacity can be updated. A fleet cannot be updated in a status of STARTING or STOPPING .
    See also: AWS API Documentation
    
    
    :example: response = client.update_fleet(
        ImageName='string',
        Name='string',
        InstanceType='string',
        ComputeCapacity={
            'DesiredInstances': 123
        },
        VpcConfig={
            'SubnetIds': [
                'string',
            ]
        },
        MaxUserDurationInSeconds=123,
        DisconnectTimeoutInSeconds=123,
        DeleteVpcConfig=True|False,
        Description='string',
        DisplayName='string',
        EnableDefaultInternetAccess=True|False
    )
    
    
    :type ImageName: string
    :param ImageName: The image name from which a fleet is created.

    :type Name: string
    :param Name: [REQUIRED]
            The name of the fleet.
            

    :type InstanceType: string
    :param InstanceType: The instance type of compute resources for the fleet. Fleet instances are launched from this instance type.

    :type ComputeCapacity: dict
    :param ComputeCapacity: The parameters for the capacity allocated to the fleet.
            DesiredInstances (integer) -- [REQUIRED]The desired number of streaming instances.
            

    :type VpcConfig: dict
    :param VpcConfig: The VPC configuration for the fleet.
            SubnetIds (list) --The list of subnets to which a network interface is established from the fleet instance.
            (string) --
            

    :type MaxUserDurationInSeconds: integer
    :param MaxUserDurationInSeconds: The maximum time for which a streaming session can run. The input can be any numeric value in seconds between 600 and 57600.

    :type DisconnectTimeoutInSeconds: integer
    :param DisconnectTimeoutInSeconds: The time after disconnection when a session is considered to have ended. If a user who got disconnected reconnects within this timeout interval, the user is connected back to their previous session. The input can be any numeric value in seconds between 60 and 57600.

    :type DeleteVpcConfig: boolean
    :param DeleteVpcConfig: Delete the VPC association for the specified fleet.

    :type Description: string
    :param Description: The description displayed to end users on the AppStream 2.0 portal.

    :type DisplayName: string
    :param DisplayName: The name displayed to end users on the AppStream 2.0 portal.

    :type EnableDefaultInternetAccess: boolean
    :param EnableDefaultInternetAccess: Enables or disables default Internet access for the fleet.

    :rtype: dict
    :return: {
        'Fleet': {
            'Arn': 'string',
            'Name': 'string',
            'DisplayName': 'string',
            'Description': 'string',
            'ImageName': 'string',
            'InstanceType': 'string',
            'ComputeCapacityStatus': {
                'Desired': 123,
                'Running': 123,
                'InUse': 123,
                'Available': 123
            },
            'MaxUserDurationInSeconds': 123,
            'DisconnectTimeoutInSeconds': 123,
            'State': 'STARTING'|'RUNNING'|'STOPPING'|'STOPPED',
            'VpcConfig': {
                'SubnetIds': [
                    'string',
                ]
            },
            'CreatedTime': datetime(2015, 1, 1),
            'FleetErrors': [
                {
                    'ErrorCode': 'IAM_SERVICE_ROLE_MISSING_ENI_DESCRIBE_ACTION'|'IAM_SERVICE_ROLE_MISSING_ENI_CREATE_ACTION'|'IAM_SERVICE_ROLE_MISSING_ENI_DELETE_ACTION'|'NETWORK_INTERFACE_LIMIT_EXCEEDED'|'INTERNAL_SERVICE_ERROR'|'IAM_SERVICE_ROLE_IS_MISSING'|'SUBNET_HAS_INSUFFICIENT_IP_ADDRESSES'|'IAM_SERVICE_ROLE_MISSING_DESCRIBE_SUBNET_ACTION'|'SUBNET_NOT_FOUND'|'IMAGE_NOT_FOUND'|'INVALID_SUBNET_CONFIGURATION',
                    'ErrorMessage': 'string'
                },
            ],
            'EnableDefaultInternetAccess': True|False
        }
    }
    
    
    :returns: 
    (string) --
    
    """
    pass