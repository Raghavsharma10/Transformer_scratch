def describe_reserved_db_instances(ReservedDBInstanceId=None, ReservedDBInstancesOfferingId=None, DBInstanceClass=None, Duration=None, ProductDescription=None, OfferingType=None, MultiAZ=None, Filters=None, MaxRecords=None, Marker=None):
    """
    Returns information about reserved DB instances for this account, or about a specified reserved DB instance.
    See also: AWS API Documentation
    
    Examples
    This example lists information for all reserved DB instances for the specified DB instance class, duration, product, offering type, and availability zone settings.
    Expected Output:
    
    :example: response = client.describe_reserved_db_instances(
        ReservedDBInstanceId='string',
        ReservedDBInstancesOfferingId='string',
        DBInstanceClass='string',
        Duration='string',
        ProductDescription='string',
        OfferingType='string',
        MultiAZ=True|False,
        Filters=[
            {
                'Name': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        MaxRecords=123,
        Marker='string'
    )
    
    
    :type ReservedDBInstanceId: string
    :param ReservedDBInstanceId: The reserved DB instance identifier filter value. Specify this parameter to show only the reservation that matches the specified reservation ID.

    :type ReservedDBInstancesOfferingId: string
    :param ReservedDBInstancesOfferingId: The offering identifier filter value. Specify this parameter to show only purchased reservations matching the specified offering identifier.

    :type DBInstanceClass: string
    :param DBInstanceClass: The DB instance class filter value. Specify this parameter to show only those reservations matching the specified DB instances class.

    :type Duration: string
    :param Duration: The duration filter value, specified in years or seconds. Specify this parameter to show only reservations for this duration.
            Valid Values: 1 | 3 | 31536000 | 94608000
            

    :type ProductDescription: string
    :param ProductDescription: The product description filter value. Specify this parameter to show only those reservations matching the specified product description.

    :type OfferingType: string
    :param OfferingType: The offering type filter value. Specify this parameter to show only the available offerings matching the specified offering type.
            Valid Values: 'Partial Upfront' | 'All Upfront' | 'No Upfront'
            

    :type MultiAZ: boolean
    :param MultiAZ: The Multi-AZ filter value. Specify this parameter to show only those reservations matching the specified Multi-AZ parameter.

    :type Filters: list
    :param Filters: This parameter is not currently supported.
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

    :rtype: dict
    :return: {
        'Marker': 'string',
        'ReservedDBInstances': [
            {
                'ReservedDBInstanceId': 'string',
                'ReservedDBInstancesOfferingId': 'string',
                'DBInstanceClass': 'string',
                'StartTime': datetime(2015, 1, 1),
                'Duration': 123,
                'FixedPrice': 123.0,
                'UsagePrice': 123.0,
                'CurrencyCode': 'string',
                'DBInstanceCount': 123,
                'ProductDescription': 'string',
                'OfferingType': 'string',
                'MultiAZ': True|False,
                'State': 'string',
                'RecurringCharges': [
                    {
                        'RecurringChargeAmount': 123.0,
                        'RecurringChargeFrequency': 'string'
                    },
                ],
                'ReservedDBInstanceArn': 'string'
            },
        ]
    }
    
    
    """
    pass