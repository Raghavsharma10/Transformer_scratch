def create_fleet(Name=None, Description=None, BuildId=None, ServerLaunchPath=None, ServerLaunchParameters=None, LogPaths=None, EC2InstanceType=None, EC2InboundPermissions=None, NewGameSessionProtectionPolicy=None, RuntimeConfiguration=None, ResourceCreationLimitPolicy=None, MetricGroups=None):
    """
    Creates a new fleet to run your game servers. A fleet is a set of Amazon Elastic Compute Cloud (Amazon EC2) instances, each of which can run multiple server processes to host game sessions. You configure a fleet to create instances with certain hardware specifications (see Amazon EC2 Instance Types for more information), and deploy a specified game build to each instance. A newly created fleet passes through several statuses; once it reaches the ACTIVE status, it can begin hosting game sessions.
    To create a new fleet, you must specify the following: (1) fleet name, (2) build ID of an uploaded game build, (3) an EC2 instance type, and (4) a runtime configuration that describes which server processes to run on each instance in the fleet. (Although the runtime configuration is not a required parameter, the fleet cannot be successfully created without it.) You can also configure the new fleet with the following settings: fleet description, access permissions for inbound traffic, fleet-wide game session protection, and resource creation limit. If you use Amazon CloudWatch for metrics, you can add the new fleet to a metric group, which allows you to view aggregated metrics for a set of fleets. Once you specify a metric group, the new fleet's metrics are included in the metric group's data.
    If the CreateFleet call is successful, Amazon GameLift performs the following tasks:
    After a fleet is created, use the following actions to change fleet properties and configuration:
    See also: AWS API Documentation
    
    
    :example: response = client.create_fleet(
        Name='string',
        Description='string',
        BuildId='string',
        ServerLaunchPath='string',
        ServerLaunchParameters='string',
        LogPaths=[
            'string',
        ],
        EC2InstanceType='t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge',
        EC2InboundPermissions=[
            {
                'FromPort': 123,
                'ToPort': 123,
                'IpRange': 'string',
                'Protocol': 'TCP'|'UDP'
            },
        ],
        NewGameSessionProtectionPolicy='NoProtection'|'FullProtection',
        RuntimeConfiguration={
            'ServerProcesses': [
                {
                    'LaunchPath': 'string',
                    'Parameters': 'string',
                    'ConcurrentExecutions': 123
                },
            ],
            'MaxConcurrentGameSessionActivations': 123,
            'GameSessionActivationTimeoutSeconds': 123
        },
        ResourceCreationLimitPolicy={
            'NewGameSessionsPerCreator': 123,
            'PolicyPeriodInMinutes': 123
        },
        MetricGroups=[
            'string',
        ]
    )
    
    
    :type Name: string
    :param Name: [REQUIRED]
            Descriptive label that is associated with a fleet. Fleet names do not need to be unique.
            

    :type Description: string
    :param Description: Human-readable description of a fleet.

    :type BuildId: string
    :param BuildId: [REQUIRED]
            Unique identifier for a build to be deployed on the new fleet. The build must have been successfully uploaded to Amazon GameLift and be in a READY status. This fleet setting cannot be changed once the fleet is created.
            

    :type ServerLaunchPath: string
    :param ServerLaunchPath: This parameter is no longer used. Instead, specify a server launch path using the RuntimeConfiguration parameter. (Requests that specify a server launch path and launch parameters instead of a runtime configuration will continue to work.)

    :type ServerLaunchParameters: string
    :param ServerLaunchParameters: This parameter is no longer used. Instead, specify server launch parameters in the RuntimeConfiguration parameter. (Requests that specify a server launch path and launch parameters instead of a runtime configuration will continue to work.)

    :type LogPaths: list
    :param LogPaths: This parameter is no longer used. Instead, to specify where Amazon GameLift should store log files once a server process shuts down, use the Amazon GameLift server API ProcessReady() and specify one or more directory paths in logParameters . See more information in the Server API Reference .
            (string) --
            

    :type EC2InstanceType: string
    :param EC2InstanceType: [REQUIRED]
            Name of an EC2 instance type that is supported in Amazon GameLift. A fleet instance type determines the computing resources of each instance in the fleet, including CPU, memory, storage, and networking capacity. Amazon GameLift supports the following EC2 instance types. See Amazon EC2 Instance Types for detailed descriptions.
            

    :type EC2InboundPermissions: list
    :param EC2InboundPermissions: Range of IP addresses and port settings that permit inbound traffic to access server processes running on the fleet. If no inbound permissions are set, including both IP address range and port range, the server processes in the fleet cannot accept connections. You can specify one or more sets of permissions for a fleet.
            (dict) --A range of IP addresses and port settings that allow inbound traffic to connect to server processes on Amazon GameLift. Each game session hosted on a fleet is assigned a unique combination of IP address and port number, which must fall into the fleet's allowed ranges. This combination is included in the GameSession object.
            FromPort (integer) -- [REQUIRED]Starting value for a range of allowed port numbers.
            ToPort (integer) -- [REQUIRED]Ending value for a range of allowed port numbers. Port numbers are end-inclusive. This value must be higher than FromPort .
            IpRange (string) -- [REQUIRED]Range of allowed IP addresses. This value must be expressed in CIDR notation. Example: '000.000.000.000/[subnet mask] ' or optionally the shortened version '0.0.0.0/[subnet mask] '.
            Protocol (string) -- [REQUIRED]Network communication protocol used by the fleet.
            
            

    :type NewGameSessionProtectionPolicy: string
    :param NewGameSessionProtectionPolicy: Game session protection policy to apply to all instances in this fleet. If this parameter is not set, instances in this fleet default to no protection. You can change a fleet's protection policy using UpdateFleetAttributes, but this change will only affect sessions created after the policy change. You can also set protection for individual instances using UpdateGameSession .
            NoProtection   The game session can be terminated during a scale-down event.
            FullProtection   If the game session is in an ACTIVE status, it cannot be terminated during a scale-down event.
            

    :type RuntimeConfiguration: dict
    :param RuntimeConfiguration: Instructions for launching server processes on each instance in the fleet. The runtime configuration for a fleet has a collection of server process configurations, one for each type of server process to run on an instance. A server process configuration specifies the location of the server executable, launch parameters, and the number of concurrent processes with that configuration to maintain on each instance. A CreateFleet request must include a runtime configuration with at least one server process configuration; otherwise the request will fail with an invalid request exception. (This parameter replaces the parameters ServerLaunchPath and ServerLaunchParameters ; requests that contain values for these parameters instead of a runtime configuration will continue to work.)
            ServerProcesses (list) --Collection of server process configurations that describe which server processes to run on each instance in a fleet.
            (dict) --A set of instructions for launching server processes on each instance in a fleet. Each instruction set identifies the location of the server executable, optional launch parameters, and the number of server processes with this configuration to maintain concurrently on the instance. Server process configurations make up a fleet's `` RuntimeConfiguration `` .
            LaunchPath (string) -- [REQUIRED]Location of the server executable in a game build. All game builds are installed on instances at the root : for Windows instances C:\game , and for Linux instances /local/game . A Windows game build with an executable file located at MyGame\latest\server.exe must have a launch path of 'C:\game\MyGame\latest\server.exe '. A Linux game build with an executable file located at MyGame/latest/server.exe must have a launch path of '/local/game/MyGame/latest/server.exe '.
            Parameters (string) --Optional list of parameters to pass to the server executable on launch.
            ConcurrentExecutions (integer) -- [REQUIRED]Number of server processes using this configuration to run concurrently on an instance.
            
            MaxConcurrentGameSessionActivations (integer) --Maximum number of game sessions with status ACTIVATING to allow on an instance simultaneously. This setting limits the amount of instance resources that can be used for new game activations at any one time.
            GameSessionActivationTimeoutSeconds (integer) --Maximum amount of time (in seconds) that a game session can remain in status ACTIVATING. If the game session is not active before the timeout, activation is terminated and the game session status is changed to TERMINATED.
            

    :type ResourceCreationLimitPolicy: dict
    :param ResourceCreationLimitPolicy: Policy that limits the number of game sessions an individual player can create over a span of time for this fleet.
            NewGameSessionsPerCreator (integer) --Maximum number of game sessions that an individual can create during the policy period.
            PolicyPeriodInMinutes (integer) --Time span used in evaluating the resource creation limit policy.
            

    :type MetricGroups: list
    :param MetricGroups: Names of metric groups to add this fleet to. Use an existing metric group name to add this fleet to the group, or use a new name to create a new metric group. Currently, a fleet can only be included in one metric group at a time.
            (string) --
            

    :rtype: dict
    :return: {
        'FleetAttributes': {
            'FleetId': 'string',
            'FleetArn': 'string',
            'Description': 'string',
            'Name': 'string',
            'CreationTime': datetime(2015, 1, 1),
            'TerminationTime': datetime(2015, 1, 1),
            'Status': 'NEW'|'DOWNLOADING'|'VALIDATING'|'BUILDING'|'ACTIVATING'|'ACTIVE'|'DELETING'|'ERROR'|'TERMINATED',
            'BuildId': 'string',
            'ServerLaunchPath': 'string',
            'ServerLaunchParameters': 'string',
            'LogPaths': [
                'string',
            ],
            'NewGameSessionProtectionPolicy': 'NoProtection'|'FullProtection',
            'OperatingSystem': 'WINDOWS_2012'|'AMAZON_LINUX',
            'ResourceCreationLimitPolicy': {
                'NewGameSessionsPerCreator': 123,
                'PolicyPeriodInMinutes': 123
            },
            'MetricGroups': [
                'string',
            ]
        }
    }
    
    
    :returns: 
    UpdateFleetAttributes -- Update fleet metadata, including name and description.
    UpdateFleetCapacity -- Increase or decrease the number of instances you want the fleet to maintain.
    UpdateFleetPortSettings -- Change the IP address and port ranges that allow access to incoming traffic.
    UpdateRuntimeConfiguration -- Change how server processes are launched in the fleet, including launch path, launch parameters, and the number of concurrent processes.
    PutScalingPolicy -- Create or update rules that are used to set the fleet's capacity (autoscaling).
    
    """
    pass