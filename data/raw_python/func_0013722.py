def describe_spot_price_history(DryRun=None, StartTime=None, EndTime=None, InstanceTypes=None, ProductDescriptions=None, Filters=None, AvailabilityZone=None, MaxResults=None, NextToken=None):
    """
    Describes the Spot price history. For more information, see Spot Instance Pricing History in the Amazon Elastic Compute Cloud User Guide .
    When you specify a start and end time, this operation returns the prices of the instance types within the time range that you specified and the time when the price changed. The price is valid within the time period that you specified; the response merely indicates the last time that the price changed.
    See also: AWS API Documentation
    
    Examples
    This example returns the Spot Price history for m1.xlarge, Linux/UNIX (Amazon VPC) instances for a particular day in January.
    Expected Output:
    
    :example: response = client.describe_spot_price_history(
        DryRun=True|False,
        StartTime=datetime(2015, 1, 1),
        EndTime=datetime(2015, 1, 1),
        InstanceTypes=[
            't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'f1.2xlarge'|'f1.16xlarge',
        ],
        ProductDescriptions=[
            'string',
        ],
        Filters=[
            {
                'Name': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        AvailabilityZone='string',
        MaxResults=123,
        NextToken='string'
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type StartTime: datetime
    :param StartTime: The date and time, up to the past 90 days, from which to start retrieving the price history data, in UTC format (for example, YYYY -MM -DD T*HH* :MM :SS Z).

    :type EndTime: datetime
    :param EndTime: The date and time, up to the current date, from which to stop retrieving the price history data, in UTC format (for example, YYYY -MM -DD T*HH* :MM :SS Z).

    :type InstanceTypes: list
    :param InstanceTypes: Filters the results by the specified instance types. Note that T2 and HS1 instance types are not supported.
            (string) --
            

    :type ProductDescriptions: list
    :param ProductDescriptions: Filters the results by the specified basic product descriptions.
            (string) --
            

    :type Filters: list
    :param Filters: One or more filters.
            availability-zone - The Availability Zone for which prices should be returned.
            instance-type - The type of instance (for example, m3.medium ).
            product-description - The product description for the Spot price (Linux/UNIX | SUSE Linux | Windows | Linux/UNIX (Amazon VPC) | SUSE Linux (Amazon VPC) | Windows (Amazon VPC) ).
            spot-price - The Spot price. The value must match exactly (or use wildcards; greater than or less than comparison is not supported).
            timestamp - The timestamp of the Spot price history, in UTC format (for example, YYYY -MM -DD T*HH* :MM :SS Z). You can use wildcards (* and ?). Greater than or less than comparison is not supported.
            (dict) --A filter name and value pair that is used to return a more specific list of results. Filters can be used to match a set of resources by various criteria, such as tags, attributes, or IDs.
            Name (string) --The name of the filter. Filter names are case-sensitive.
            Values (list) --One or more filter values. Filter values are case-sensitive.
            (string) --
            
            

    :type AvailabilityZone: string
    :param AvailabilityZone: Filters the results by the specified Availability Zone.

    :type MaxResults: integer
    :param MaxResults: The maximum number of results to return in a single call. Specify a value between 1 and 1000. The default value is 1000. To retrieve the remaining results, make another call with the returned NextToken value.

    :type NextToken: string
    :param NextToken: The token for the next set of results.

    :rtype: dict
    :return: {
        'SpotPriceHistory': [
            {
                'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'f1.2xlarge'|'f1.16xlarge',
                'ProductDescription': 'Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
                'SpotPrice': 'string',
                'Timestamp': datetime(2015, 1, 1),
                'AvailabilityZone': 'string'
            },
        ],
        'NextToken': 'string'
    }
    
    
    """
    pass