def describe_users(OrganizationId=None, UserIds=None, Query=None, Include=None, Order=None, Sort=None, Marker=None, Limit=None, Fields=None):
    """
    Describes the specified users. You can describe all users or filter the results (for example, by status or organization).
    By default, Amazon WorkDocs returns the first 24 active or pending users. If there are more results, the response includes a marker that you can use to request the next set of results.
    See also: AWS API Documentation
    
    
    :example: response = client.describe_users(
        OrganizationId='string',
        UserIds='string',
        Query='string',
        Include='ALL'|'ACTIVE_PENDING',
        Order='ASCENDING'|'DESCENDING',
        Sort='USER_NAME'|'FULL_NAME'|'STORAGE_LIMIT'|'USER_STATUS'|'STORAGE_USED',
        Marker='string',
        Limit=123,
        Fields='string'
    )
    
    
    :type OrganizationId: string
    :param OrganizationId: The ID of the organization.

    :type UserIds: string
    :param UserIds: The IDs of the users.

    :type Query: string
    :param Query: A query to filter users by user name.

    :type Include: string
    :param Include: The state of the users. Specify 'ALL' to include inactive users.

    :type Order: string
    :param Order: The order for the results.

    :type Sort: string
    :param Sort: The sorting criteria.

    :type Marker: string
    :param Marker: The marker for the next set of results. (You received this marker from a previous call.)

    :type Limit: integer
    :param Limit: The maximum number of items to return.

    :type Fields: string
    :param Fields: A comma-separated list of values. Specify 'STORAGE_METADATA' to include the user storage quota and utilization information.

    :rtype: dict
    :return: {
        'Users': [
            {
                'Id': 'string',
                'Username': 'string',
                'EmailAddress': 'string',
                'GivenName': 'string',
                'Surname': 'string',
                'OrganizationId': 'string',
                'RootFolderId': 'string',
                'RecycleBinFolderId': 'string',
                'Status': 'ACTIVE'|'INACTIVE'|'PENDING',
                'Type': 'USER'|'ADMIN',
                'CreatedTimestamp': datetime(2015, 1, 1),
                'ModifiedTimestamp': datetime(2015, 1, 1),
                'TimeZoneId': 'string',
                'Locale': 'en'|'fr'|'ko'|'de'|'es'|'ja'|'ru'|'zh_CN'|'zh_TW'|'pt_BR'|'default',
                'Storage': {
                    'StorageUtilizedInBytes': 123,
                    'StorageRule': {
                        'StorageAllocatedInBytes': 123,
                        'StorageType': 'UNLIMITED'|'QUOTA'
                    }
                }
            },
        ],
        'TotalNumberOfUsers': 123,
        'Marker': 'string'
    }
    
    
    """
    pass