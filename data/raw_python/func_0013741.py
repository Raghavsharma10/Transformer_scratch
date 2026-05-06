def create_server(AssociatePublicIpAddress=None, DisableAutomatedBackup=None, Engine=None, EngineModel=None, EngineVersion=None, EngineAttributes=None, BackupRetentionCount=None, ServerName=None, InstanceProfileArn=None, InstanceType=None, KeyPair=None, PreferredMaintenanceWindow=None, PreferredBackupWindow=None, SecurityGroupIds=None, ServiceRoleArn=None, SubnetIds=None, BackupId=None):
    """
    Creates and immedately starts a new server. The server is ready to use when it is in the HEALTHY state. By default, you can create a maximum of 10 servers.
    This operation is asynchronous.
    A LimitExceededException is thrown when you have created the maximum number of servers (10). A ResourceAlreadyExistsException is thrown when a server with the same name already exists in the account. A ResourceNotFoundException is thrown when you specify a backup ID that is not valid or is for a backup that does not exist. A ValidationException is thrown when parameters of the request are not valid.
    If you do not specify a security group by adding the SecurityGroupIds parameter, AWS OpsWorks creates a new security group. The default security group opens the Chef server to the world on TCP port 443. If a KeyName is present, AWS OpsWorks enables SSH access. SSH is also open to the world on TCP port 22.
    By default, the Chef Server is accessible from any IP address. We recommend that you update your security group rules to allow access from known IP addresses and address ranges only. To edit security group rules, open Security Groups in the navigation pane of the EC2 management console.
    See also: AWS API Documentation
    
    
    :example: response = client.create_server(
        AssociatePublicIpAddress=True|False,
        DisableAutomatedBackup=True|False,
        Engine='string',
        EngineModel='string',
        EngineVersion='string',
        EngineAttributes=[
            {
                'Name': 'string',
                'Value': 'string'
            },
        ],
        BackupRetentionCount=123,
        ServerName='string',
        InstanceProfileArn='string',
        InstanceType='string',
        KeyPair='string',
        PreferredMaintenanceWindow='string',
        PreferredBackupWindow='string',
        SecurityGroupIds=[
            'string',
        ],
        ServiceRoleArn='string',
        SubnetIds=[
            'string',
        ],
        BackupId='string'
    )
    
    
    :type AssociatePublicIpAddress: boolean
    :param AssociatePublicIpAddress: Associate a public IP address with a server that you are launching. Valid values are true or false . The default value is true .

    :type DisableAutomatedBackup: boolean
    :param DisableAutomatedBackup: Enable or disable scheduled backups. Valid values are true or false . The default value is true .

    :type Engine: string
    :param Engine: The configuration management engine to use. Valid values include Chef .

    :type EngineModel: string
    :param EngineModel: The engine model, or option. Valid values include Single .

    :type EngineVersion: string
    :param EngineVersion: The major release version of the engine that you want to use. Values depend on the engine that you choose.

    :type EngineAttributes: list
    :param EngineAttributes: Optional engine attributes on a specified server.
            Attributes accepted in a createServer request:
            CHEF_PIVOTAL_KEY : A base64-encoded RSA private key that is not stored by AWS OpsWorks for Chef. This private key is required to access the Chef API. When no CHEF_PIVOTAL_KEY is set, one is generated and returned in the response.
            CHEF_DELIVERY_ADMIN_PASSWORD : The password for the administrative user in the Chef Automate GUI. The password length is a minimum of eight characters, and a maximum of 32. The password can contain letters, numbers, and special characters (!/@#$%^+=_). The password must contain at least one lower case letter, one upper case letter, one number, and one special character. When no CHEF_DELIVERY_ADMIN_PASSWORD is set, one is generated and returned in the response.
            (dict) --A name and value pair that is specific to the engine of the server.
            Name (string) --The name of the engine attribute.
            Value (string) --The value of the engine attribute.
            
            

    :type BackupRetentionCount: integer
    :param BackupRetentionCount: The number of automated backups that you want to keep. Whenever a new backup is created, AWS OpsWorks for Chef Automate deletes the oldest backups if this number is exceeded. The default value is 1 .

    :type ServerName: string
    :param ServerName: [REQUIRED]
            The name of the server. The server name must be unique within your AWS account, within each region. Server names must start with a letter; then letters, numbers, or hyphens (-) are allowed, up to a maximum of 40 characters.
            

    :type InstanceProfileArn: string
    :param InstanceProfileArn: [REQUIRED]
            The ARN of the instance profile that your Amazon EC2 instances use. Although the AWS OpsWorks console typically creates the instance profile for you, if you are using API commands instead, run the service-role-creation.yaml AWS CloudFormation template, located at https://s3.amazonaws.com/opsworks-cm-us-east-1-prod-default-assets/misc/opsworks-cm-roles.yaml. This template creates a CloudFormation stack that includes the instance profile you need.
            

    :type InstanceType: string
    :param InstanceType: [REQUIRED]
            The Amazon EC2 instance type to use. Valid values must be specified in the following format: ^([cm][34]|t2).* For example, m4.large . Valid values are t2.medium , m4.large , or m4.2xlarge .
            

    :type KeyPair: string
    :param KeyPair: The Amazon EC2 key pair to set for the instance. This parameter is optional; if desired, you may specify this parameter to connect to your instances by using SSH.

    :type PreferredMaintenanceWindow: string
    :param PreferredMaintenanceWindow: The start time for a one-hour period each week during which AWS OpsWorks for Chef Automate performs maintenance on the instance. Valid values must be specified in the following format: DDD:HH:MM . The specified time is in coordinated universal time (UTC). The default value is a random one-hour period on Tuesday, Wednesday, or Friday. See TimeWindowDefinition for more information.
            Example: Mon:08:00 , which represents a start time of every Monday at 08:00 UTC. (8:00 a.m.)
            

    :type PreferredBackupWindow: string
    :param PreferredBackupWindow: The start time for a one-hour period during which AWS OpsWorks for Chef Automate backs up application-level data on your server if automated backups are enabled. Valid values must be specified in one of the following formats:
            HH:MM for daily backups
            DDD:HH:MM for weekly backups
            The specified time is in coordinated universal time (UTC). The default value is a random, daily start time.
            Example: 08:00 , which represents a daily start time of 08:00 UTC.Example: Mon:08:00 , which represents a start time of every Monday at 08:00 UTC. (8:00 a.m.)
            

    :type SecurityGroupIds: list
    :param SecurityGroupIds: A list of security group IDs to attach to the Amazon EC2 instance. If you add this parameter, the specified security groups must be within the VPC that is specified by SubnetIds .
            If you do not specify this parameter, AWS OpsWorks for Chef Automate creates one new security group that uses TCP ports 22 and 443, open to 0.0.0.0/0 (everyone).
            (string) --
            

    :type ServiceRoleArn: string
    :param ServiceRoleArn: [REQUIRED]
            The service role that the AWS OpsWorks for Chef Automate service backend uses to work with your account. Although the AWS OpsWorks management console typically creates the service role for you, if you are using the AWS CLI or API commands, run the service-role-creation.yaml AWS CloudFormation template, located at https://s3.amazonaws.com/opsworks-stuff/latest/service-role-creation.yaml. This template creates a CloudFormation stack that includes the service role that you need.
            

    :type SubnetIds: list
    :param SubnetIds: The IDs of subnets in which to launch the server EC2 instance.
            Amazon EC2-Classic customers: This field is required. All servers must run within a VPC. The VPC must have 'Auto Assign Public IP' enabled.
            EC2-VPC customers: This field is optional. If you do not specify subnet IDs, your EC2 instances are created in a default subnet that is selected by Amazon EC2. If you specify subnet IDs, the VPC must have 'Auto Assign Public IP' enabled.
            For more information about supported Amazon EC2 platforms, see Supported Platforms .
            (string) --
            

    :type BackupId: string
    :param BackupId: If you specify this field, AWS OpsWorks for Chef Automate creates the server by using the backup represented by BackupId.

    :rtype: dict
    :return: {
        'Server': {
            'AssociatePublicIpAddress': True|False,
            'BackupRetentionCount': 123,
            'ServerName': 'string',
            'CreatedAt': datetime(2015, 1, 1),
            'CloudFormationStackArn': 'string',
            'DisableAutomatedBackup': True|False,
            'Endpoint': 'string',
            'Engine': 'string',
            'EngineModel': 'string',
            'EngineAttributes': [
                {
                    'Name': 'string',
                    'Value': 'string'
                },
            ],
            'EngineVersion': 'string',
            'InstanceProfileArn': 'string',
            'InstanceType': 'string',
            'KeyPair': 'string',
            'MaintenanceStatus': 'SUCCESS'|'FAILED',
            'PreferredMaintenanceWindow': 'string',
            'PreferredBackupWindow': 'string',
            'SecurityGroupIds': [
                'string',
            ],
            'ServiceRoleArn': 'string',
            'Status': 'BACKING_UP'|'CONNECTION_LOST'|'CREATING'|'DELETING'|'MODIFYING'|'FAILED'|'HEALTHY'|'RUNNING'|'RESTORING'|'SETUP'|'UNDER_MAINTENANCE'|'UNHEALTHY'|'TERMINATED',
            'StatusReason': 'string',
            'SubnetIds': [
                'string',
            ],
            'ServerArn': 'string'
        }
    }
    
    
    :returns: 
    CHEF_PIVOTAL_KEY : A base64-encoded RSA private key that is generated by AWS OpsWorks for Chef Automate. This private key is required to access the Chef API.
    CHEF_STARTER_KIT : A base64-encoded ZIP file. The ZIP file contains a Chef starter kit, which includes a README, a configuration file, and the required RSA private key. Save this file, unzip it, and then change to the directory where you've unzipped the file contents. From this directory, you can run Knife commands.
    
    """
    pass