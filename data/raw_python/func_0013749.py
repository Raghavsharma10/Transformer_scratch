def describe_db_engine_versions(Engine=None, EngineVersion=None, DBParameterGroupFamily=None, Filters=None, MaxRecords=None, Marker=None, DefaultOnly=None, ListSupportedCharacterSets=None, ListSupportedTimezones=None):
    """
    Returns a list of the available DB engines.
    See also: AWS API Documentation
    
    Examples
    This example lists settings for the specified DB engine version.
    Expected Output:
    
    :example: response = client.describe_db_engine_versions(
        Engine='string',
        EngineVersion='string',
        DBParameterGroupFamily='string',
        Filters=[
            {
                'Name': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        MaxRecords=123,
        Marker='string',
        DefaultOnly=True|False,
        ListSupportedCharacterSets=True|False,
        ListSupportedTimezones=True|False
    )
    
    
    :type Engine: string
    :param Engine: The database engine to return.

    :type EngineVersion: string
    :param EngineVersion: The database engine version to return.
            Example: 5.1.49
            

    :type DBParameterGroupFamily: string
    :param DBParameterGroupFamily: The name of a specific DB parameter group family to return details for.
            Constraints:
            Must be 1 to 255 alphanumeric characters
            First character must be a letter
            Cannot end with a hyphen or contain two consecutive hyphens
            

    :type Filters: list
    :param Filters: Not currently supported.
            (dict) --This type is not currently supported.
            Name (string) -- [REQUIRED]This parameter is not currently supported.
            Values (list) -- [REQUIRED]This parameter is not currently supported.
            (string) --
            
            

    :type MaxRecords: integer
    :param MaxRecords: The maximum number of records to include in the response. If more than the MaxRecords value is available, a pagination token called a marker is included in the response so that the following results can be retrieved.
            Default: 100
            Constraints: Minimum 20, maximum 100.
            

    :type Marker: string
    :param Marker: An optional pagination token provided by a previous request. If this parameter is specified, the response includes only records beyond the marker, up to the value specified by MaxRecords .

    :type DefaultOnly: boolean
    :param DefaultOnly: Indicates that only the default version of the specified engine or engine and major version combination is returned.

    :type ListSupportedCharacterSets: boolean
    :param ListSupportedCharacterSets: If this parameter is specified and the requested engine supports the CharacterSetName parameter for CreateDBInstance , the response includes a list of supported character sets for each engine version.

    :type ListSupportedTimezones: boolean
    :param ListSupportedTimezones: If this parameter is specified and the requested engine supports the TimeZone parameter for CreateDBInstance , the response includes a list of supported time zones for each engine version.

    :rtype: dict
    :return: {
        'Marker': 'string',
        'DBEngineVersions': [
            {
                'Engine': 'string',
                'EngineVersion': 'string',
                'DBParameterGroupFamily': 'string',
                'DBEngineDescription': 'string',
                'DBEngineVersionDescription': 'string',
                'DefaultCharacterSet': {
                    'CharacterSetName': 'string',
                    'CharacterSetDescription': 'string'
                },
                'SupportedCharacterSets': [
                    {
                        'CharacterSetName': 'string',
                        'CharacterSetDescription': 'string'
                    },
                ],
                'ValidUpgradeTarget': [
                    {
                        'Engine': 'string',
                        'EngineVersion': 'string',
                        'Description': 'string',
                        'AutoUpgrade': True|False,
                        'IsMajorVersionUpgrade': True|False
                    },
                ],
                'SupportedTimezones': [
                    {
                        'TimezoneName': 'string'
                    },
                ]
            },
        ]
    }
    
    
    """
    pass