def describe_reserved_instances_offerings(DryRun=None, ReservedInstancesOfferingIds=None, InstanceType=None, AvailabilityZone=None, ProductDescription=None, Filters=None, InstanceTenancy=None, OfferingType=None, NextToken=None, MaxResults=None, IncludeMarketplace=None, MinDuration=None, MaxDuration=None, MaxInstanceCount=None, OfferingClass=None):
    """
    Describes Reserved Instance offerings that are available for purchase. With Reserved Instances, you purchase the right to launch instances for a period of time. During that time period, you do not receive insufficient capacity errors, and you pay a lower usage rate than the rate charged for On-Demand instances for the actual time used.
    If you have listed your own Reserved Instances for sale in the Reserved Instance Marketplace, they will be excluded from these results. This is to ensure that you do not purchase your own Reserved Instances.
    For more information, see Reserved Instance Marketplace in the Amazon Elastic Compute Cloud User Guide .
    See also: AWS API Documentation
    
    
    :example: response = client.describe_reserved_instances_offerings(
        DryRun=True|False,
        ReservedInstancesOfferingIds=[
            'string',
        ],
        InstanceType='t1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'f1.2xlarge'|'f1.16xlarge',
        AvailabilityZone='string',
        ProductDescription='Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
        Filters=[
            {
                'Name': 'string',
                'Values': [
                    'string',
                ]
            },
        ],
        InstanceTenancy='default'|'dedicated'|'host',
        OfferingType='Heavy Utilization'|'Medium Utilization'|'Light Utilization'|'No Upfront'|'Partial Upfront'|'All Upfront',
        NextToken='string',
        MaxResults=123,
        IncludeMarketplace=True|False,
        MinDuration=123,
        MaxDuration=123,
        MaxInstanceCount=123,
        OfferingClass='standard'|'convertible'
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type ReservedInstancesOfferingIds: list
    :param ReservedInstancesOfferingIds: One or more Reserved Instances offering IDs.
            (string) --
            

    :type InstanceType: string
    :param InstanceType: The instance type that the reservation will cover (for example, m1.small ). For more information, see Instance Types in the Amazon Elastic Compute Cloud User Guide .

    :type AvailabilityZone: string
    :param AvailabilityZone: The Availability Zone in which the Reserved Instance can be used.

    :type ProductDescription: string
    :param ProductDescription: The Reserved Instance product platform description. Instances that include (Amazon VPC) in the description are for use with Amazon VPC.

    :type Filters: list
    :param Filters: One or more filters.
            availability-zone - The Availability Zone where the Reserved Instance can be used.
            duration - The duration of the Reserved Instance (for example, one year or three years), in seconds (31536000 | 94608000 ).
            fixed-price - The purchase price of the Reserved Instance (for example, 9800.0).
            instance-type - The instance type that is covered by the reservation.
            marketplace - Set to true to show only Reserved Instance Marketplace offerings. When this filter is not used, which is the default behavior, all offerings from both AWS and the Reserved Instance Marketplace are listed.
            product-description - The Reserved Instance product platform description. Instances that include (Amazon VPC) in the product platform description will only be displayed to EC2-Classic account holders and are for use with Amazon VPC. (Linux/UNIX | Linux/UNIX (Amazon VPC) | SUSE Linux | SUSE Linux (Amazon VPC) | Red Hat Enterprise Linux | Red Hat Enterprise Linux (Amazon VPC) | Windows | Windows (Amazon VPC) | Windows with SQL Server Standard | Windows with SQL Server Standard (Amazon VPC) | Windows with SQL Server Web | Windows with SQL Server Web (Amazon VPC) | Windows with SQL Server Enterprise | Windows with SQL Server Enterprise (Amazon VPC) )
            reserved-instances-offering-id - The Reserved Instances offering ID.
            scope - The scope of the Reserved Instance (Availability Zone or Region ).
            usage-price - The usage price of the Reserved Instance, per hour (for example, 0.84).
            (dict) --A filter name and value pair that is used to return a more specific list of results. Filters can be used to match a set of resources by various criteria, such as tags, attributes, or IDs.
            Name (string) --The name of the filter. Filter names are case-sensitive.
            Values (list) --One or more filter values. Filter values are case-sensitive.
            (string) --
            
            

    :type InstanceTenancy: string
    :param InstanceTenancy: The tenancy of the instances covered by the reservation. A Reserved Instance with a tenancy of dedicated is applied to instances that run in a VPC on single-tenant hardware (i.e., Dedicated Instances).
            Important: The host value cannot be used with this parameter. Use the default or dedicated values only.
            Default: default
            

    :type OfferingType: string
    :param OfferingType: The Reserved Instance offering type. If you are using tools that predate the 2011-11-01 API version, you only have access to the Medium Utilization Reserved Instance offering type.

    :type NextToken: string
    :param NextToken: The token to retrieve the next page of results.

    :type MaxResults: integer
    :param MaxResults: The maximum number of results to return for the request in a single page. The remaining results of the initial request can be seen by sending another request with the returned NextToken value. The maximum is 100.
            Default: 100
            

    :type IncludeMarketplace: boolean
    :param IncludeMarketplace: Include Reserved Instance Marketplace offerings in the response.

    :type MinDuration: integer
    :param MinDuration: The minimum duration (in seconds) to filter when searching for offerings.
            Default: 2592000 (1 month)
            

    :type MaxDuration: integer
    :param MaxDuration: The maximum duration (in seconds) to filter when searching for offerings.
            Default: 94608000 (3 years)
            

    :type MaxInstanceCount: integer
    :param MaxInstanceCount: The maximum number of instances to filter when searching for offerings.
            Default: 20
            

    :type OfferingClass: string
    :param OfferingClass: The offering class of the Reserved Instance. Can be standard or convertible .

    :rtype: dict
    :return: {
        'ReservedInstancesOfferings': [
            {
                'ReservedInstancesOfferingId': 'string',
                'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'f1.2xlarge'|'f1.16xlarge',
                'AvailabilityZone': 'string',
                'Duration': 123,
                'UsagePrice': ...,
                'FixedPrice': ...,
                'ProductDescription': 'Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
                'InstanceTenancy': 'default'|'dedicated'|'host',
                'CurrencyCode': 'USD',
                'OfferingType': 'Heavy Utilization'|'Medium Utilization'|'Light Utilization'|'No Upfront'|'Partial Upfront'|'All Upfront',
                'RecurringCharges': [
                    {
                        'Frequency': 'Hourly',
                        'Amount': 123.0
                    },
                ],
                'Marketplace': True|False,
                'PricingDetails': [
                    {
                        'Price': 123.0,
                        'Count': 123
                    },
                ],
                'OfferingClass': 'standard'|'convertible',
                'Scope': 'Availability Zone'|'Region'
            },
        ],
        'NextToken': 'string'
    }
    
    
    """
    pass