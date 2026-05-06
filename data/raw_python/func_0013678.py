def create_layer(StackId=None, Type=None, Name=None, Shortname=None, Attributes=None, CloudWatchLogsConfiguration=None, CustomInstanceProfileArn=None, CustomJson=None, CustomSecurityGroupIds=None, Packages=None, VolumeConfigurations=None, EnableAutoHealing=None, AutoAssignElasticIps=None, AutoAssignPublicIps=None, CustomRecipes=None, InstallUpdatesOnBoot=None, UseEbsOptimizedInstances=None, LifecycleEventConfiguration=None):
    """
    Creates a layer. For more information, see How to Create a Layer .
    See also: AWS API Documentation
    
    
    :example: response = client.create_layer(
        StackId='string',
        Type='aws-flow-ruby'|'ecs-cluster'|'java-app'|'lb'|'web'|'php-app'|'rails-app'|'nodejs-app'|'memcached'|'db-master'|'monitoring-master'|'custom',
        Name='string',
        Shortname='string',
        Attributes={
            'string': 'string'
        },
        CloudWatchLogsConfiguration={
            'Enabled': True|False,
            'LogStreams': [
                {
                    'LogGroupName': 'string',
                    'DatetimeFormat': 'string',
                    'TimeZone': 'LOCAL'|'UTC',
                    'File': 'string',
                    'FileFingerprintLines': 'string',
                    'MultiLineStartPattern': 'string',
                    'InitialPosition': 'start_of_file'|'end_of_file',
                    'Encoding': 'ascii'|'big5'|'big5hkscs'|'cp037'|'cp424'|'cp437'|'cp500'|'cp720'|'cp737'|'cp775'|'cp850'|'cp852'|'cp855'|'cp856'|'cp857'|'cp858'|'cp860'|'cp861'|'cp862'|'cp863'|'cp864'|'cp865'|'cp866'|'cp869'|'cp874'|'cp875'|'cp932'|'cp949'|'cp950'|'cp1006'|'cp1026'|'cp1140'|'cp1250'|'cp1251'|'cp1252'|'cp1253'|'cp1254'|'cp1255'|'cp1256'|'cp1257'|'cp1258'|'euc_jp'|'euc_jis_2004'|'euc_jisx0213'|'euc_kr'|'gb2312'|'gbk'|'gb18030'|'hz'|'iso2022_jp'|'iso2022_jp_1'|'iso2022_jp_2'|'iso2022_jp_2004'|'iso2022_jp_3'|'iso2022_jp_ext'|'iso2022_kr'|'latin_1'|'iso8859_2'|'iso8859_3'|'iso8859_4'|'iso8859_5'|'iso8859_6'|'iso8859_7'|'iso8859_8'|'iso8859_9'|'iso8859_10'|'iso8859_13'|'iso8859_14'|'iso8859_15'|'iso8859_16'|'johab'|'koi8_r'|'koi8_u'|'mac_cyrillic'|'mac_greek'|'mac_iceland'|'mac_latin2'|'mac_roman'|'mac_turkish'|'ptcp154'|'shift_jis'|'shift_jis_2004'|'shift_jisx0213'|'utf_32'|'utf_32_be'|'utf_32_le'|'utf_16'|'utf_16_be'|'utf_16_le'|'utf_7'|'utf_8'|'utf_8_sig',
                    'BufferDuration': 123,
                    'BatchCount': 123,
                    'BatchSize': 123
                },
            ]
        },
        CustomInstanceProfileArn='string',
        CustomJson='string',
        CustomSecurityGroupIds=[
            'string',
        ],
        Packages=[
            'string',
        ],
        VolumeConfigurations=[
            {
                'MountPoint': 'string',
                'RaidLevel': 123,
                'NumberOfDisks': 123,
                'Size': 123,
                'VolumeType': 'string',
                'Iops': 123
            },
        ],
        EnableAutoHealing=True|False,
        AutoAssignElasticIps=True|False,
        AutoAssignPublicIps=True|False,
        CustomRecipes={
            'Setup': [
                'string',
            ],
            'Configure': [
                'string',
            ],
            'Deploy': [
                'string',
            ],
            'Undeploy': [
                'string',
            ],
            'Shutdown': [
                'string',
            ]
        },
        InstallUpdatesOnBoot=True|False,
        UseEbsOptimizedInstances=True|False,
        LifecycleEventConfiguration={
            'Shutdown': {
                'ExecutionTimeout': 123,
                'DelayUntilElbConnectionsDrained': True|False
            }
        }
    )
    
    
    :type StackId: string
    :param StackId: [REQUIRED]
            The layer stack ID.
            

    :type Type: string
    :param Type: [REQUIRED]
            The layer type. A stack cannot have more than one built-in layer of the same type. It can have any number of custom layers. Built-in layers are not available in Chef 12 stacks.
            

    :type Name: string
    :param Name: [REQUIRED]
            The layer name, which is used by the console.
            

    :type Shortname: string
    :param Shortname: [REQUIRED]
            For custom layers only, use this parameter to specify the layer's short name, which is used internally by AWS OpsWorks Stacks and by Chef recipes. The short name is also used as the name for the directory where your app files are installed. It can have a maximum of 200 characters, which are limited to the alphanumeric characters, '-', '_', and '.'.
            The built-in layers' short names are defined by AWS OpsWorks Stacks. For more information, see the Layer Reference .
            

    :type Attributes: dict
    :param Attributes: One or more user-defined key-value pairs to be added to the stack attributes.
            To create a cluster layer, set the EcsClusterArn attribute to the cluster's ARN.
            (string) --
            (string) --
            

    :type CloudWatchLogsConfiguration: dict
    :param CloudWatchLogsConfiguration: Specifies CloudWatch Logs configuration options for the layer. For more information, see CloudWatchLogsLogStream .
            Enabled (boolean) --Whether CloudWatch Logs is enabled for a layer.
            LogStreams (list) --A list of configuration options for CloudWatch Logs.
            (dict) --Describes the Amazon CloudWatch logs configuration for a layer. For detailed information about members of this data type, see the CloudWatch Logs Agent Reference .
            LogGroupName (string) --Specifies the destination log group. A log group is created automatically if it doesn't already exist. Log group names can be between 1 and 512 characters long. Allowed characters include a-z, A-Z, 0-9, '_' (underscore), '-' (hyphen), '/' (forward slash), and '.' (period).
            DatetimeFormat (string) --Specifies how the time stamp is extracted from logs. For more information, see the CloudWatch Logs Agent Reference .
            TimeZone (string) --Specifies the time zone of log event time stamps.
            File (string) --Specifies log files that you want to push to CloudWatch Logs.
            File can point to a specific file or multiple files (by using wild card characters such as /var/log/system.log* ). Only the latest file is pushed to CloudWatch Logs, based on file modification time. We recommend that you use wild card characters to specify a series of files of the same type, such as access_log.2014-06-01-01 , access_log.2014-06-01-02 , and so on by using a pattern like access_log.* . Don't use a wildcard to match multiple file types, such as access_log_80 and access_log_443 . To specify multiple, different file types, add another log stream entry to the configuration file, so that each log file type is stored in a different log group.
            Zipped files are not supported.
            FileFingerprintLines (string) --Specifies the range of lines for identifying a file. The valid values are one number, or two dash-delimited numbers, such as '1', '2-5'. The default value is '1', meaning the first line is used to calculate the fingerprint. Fingerprint lines are not sent to CloudWatch Logs unless all specified lines are available.
            MultiLineStartPattern (string) --Specifies the pattern for identifying the start of a log message.
            InitialPosition (string) --Specifies where to start to read data (start_of_file or end_of_file). The default is start_of_file. This setting is only used if there is no state persisted for that log stream.
            Encoding (string) --Specifies the encoding of the log file so that the file can be read correctly. The default is utf_8 . Encodings supported by Python codecs.decode() can be used here.
            BufferDuration (integer) --Specifies the time duration for the batching of log events. The minimum value is 5000ms and default value is 5000ms.
            BatchCount (integer) --Specifies the max number of log events in a batch, up to 10000. The default value is 1000.
            BatchSize (integer) --Specifies the maximum size of log events in a batch, in bytes, up to 1048576 bytes. The default value is 32768 bytes. This size is calculated as the sum of all event messages in UTF-8, plus 26 bytes for each log event.
            
            

    :type CustomInstanceProfileArn: string
    :param CustomInstanceProfileArn: The ARN of an IAM profile to be used for the layer's EC2 instances. For more information about IAM ARNs, see Using Identifiers .

    :type CustomJson: string
    :param CustomJson: A JSON-formatted string containing custom stack configuration and deployment attributes to be installed on the layer's instances. For more information, see Using Custom JSON . This feature is supported as of version 1.7.42 of the AWS CLI.

    :type CustomSecurityGroupIds: list
    :param CustomSecurityGroupIds: An array containing the layer custom security group IDs.
            (string) --
            

    :type Packages: list
    :param Packages: An array of Package objects that describes the layer packages.
            (string) --
            

    :type VolumeConfigurations: list
    :param VolumeConfigurations: A VolumeConfigurations object that describes the layer's Amazon EBS volumes.
            (dict) --Describes an Amazon EBS volume configuration.
            MountPoint (string) -- [REQUIRED]The volume mount point. For example '/dev/sdh'.
            RaidLevel (integer) --The volume RAID level .
            NumberOfDisks (integer) -- [REQUIRED]The number of disks in the volume.
            Size (integer) -- [REQUIRED]The volume size.
            VolumeType (string) --The volume type:
            standard - Magnetic
            io1 - Provisioned IOPS (SSD)
            gp2 - General Purpose (SSD)
            Iops (integer) --For PIOPS volumes, the IOPS per disk.
            
            

    :type EnableAutoHealing: boolean
    :param EnableAutoHealing: Whether to disable auto healing for the layer.

    :type AutoAssignElasticIps: boolean
    :param AutoAssignElasticIps: Whether to automatically assign an Elastic IP address to the layer's instances. For more information, see How to Edit a Layer .

    :type AutoAssignPublicIps: boolean
    :param AutoAssignPublicIps: For stacks that are running in a VPC, whether to automatically assign a public IP address to the layer's instances. For more information, see How to Edit a Layer .

    :type CustomRecipes: dict
    :param CustomRecipes: A LayerCustomRecipes object that specifies the layer custom recipes.
            Setup (list) --An array of custom recipe names to be run following a setup event.
            (string) --
            Configure (list) --An array of custom recipe names to be run following a configure event.
            (string) --
            Deploy (list) --An array of custom recipe names to be run following a deploy event.
            (string) --
            Undeploy (list) --An array of custom recipe names to be run following a undeploy event.
            (string) --
            Shutdown (list) --An array of custom recipe names to be run following a shutdown event.
            (string) --
            

    :type InstallUpdatesOnBoot: boolean
    :param InstallUpdatesOnBoot: Whether to install operating system and package updates when the instance boots. The default value is true . To control when updates are installed, set this value to false . You must then update your instances manually by using CreateDeployment to run the update_dependencies stack command or by manually running yum (Amazon Linux) or apt-get (Ubuntu) on the instances.
            Note
            To ensure that your instances have the latest security updates, we strongly recommend using the default value of true .
            

    :type UseEbsOptimizedInstances: boolean
    :param UseEbsOptimizedInstances: Whether to use Amazon EBS-optimized instances.

    :type LifecycleEventConfiguration: dict
    :param LifecycleEventConfiguration: A LifeCycleEventConfiguration object that you can use to configure the Shutdown event to specify an execution timeout and enable or disable Elastic Load Balancer connection draining.
            Shutdown (dict) --A ShutdownEventConfiguration object that specifies the Shutdown event configuration.
            ExecutionTimeout (integer) --The time, in seconds, that AWS OpsWorks Stacks will wait after triggering a Shutdown event before shutting down an instance.
            DelayUntilElbConnectionsDrained (boolean) --Whether to enable Elastic Load Balancing connection draining. For more information, see Connection Draining
            
            

    :rtype: dict
    :return: {
        'LayerId': 'string'
    }
    
    
    """
    pass