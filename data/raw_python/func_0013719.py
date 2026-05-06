def create_network_interface(SubnetId=None, Description=None, PrivateIpAddress=None, Groups=None, PrivateIpAddresses=None, SecondaryPrivateIpAddressCount=None, Ipv6Addresses=None, Ipv6AddressCount=None, DryRun=None):
    """
    Creates a network interface in the specified subnet.
    For more information about network interfaces, see Elastic Network Interfaces in the Amazon Virtual Private Cloud User Guide .
    See also: AWS API Documentation
    
    Examples
    This example creates a network interface for the specified subnet.
    Expected Output:
    
    :example: response = client.create_network_interface(
        SubnetId='string',
        Description='string',
        PrivateIpAddress='string',
        Groups=[
            'string',
        ],
        PrivateIpAddresses=[
            {
                'PrivateIpAddress': 'string',
                'Primary': True|False
            },
        ],
        SecondaryPrivateIpAddressCount=123,
        Ipv6Addresses=[
            {
                'Ipv6Address': 'string'
            },
        ],
        Ipv6AddressCount=123,
        DryRun=True|False
    )
    
    
    :type SubnetId: string
    :param SubnetId: [REQUIRED]
            The ID of the subnet to associate with the network interface.
            

    :type Description: string
    :param Description: A description for the network interface.

    :type PrivateIpAddress: string
    :param PrivateIpAddress: The primary private IPv4 address of the network interface. If you don't specify an IPv4 address, Amazon EC2 selects one for you from the subnet's IPv4 CIDR range. If you specify an IP address, you cannot indicate any IP addresses specified in privateIpAddresses as primary (only one IP address can be designated as primary).

    :type Groups: list
    :param Groups: The IDs of one or more security groups.
            (string) --
            

    :type PrivateIpAddresses: list
    :param PrivateIpAddresses: One or more private IPv4 addresses.
            (dict) --Describes a secondary private IPv4 address for a network interface.
            PrivateIpAddress (string) -- [REQUIRED]The private IPv4 addresses.
            Primary (boolean) --Indicates whether the private IPv4 address is the primary private IPv4 address. Only one IPv4 address can be designated as primary.
            
            

    :type SecondaryPrivateIpAddressCount: integer
    :param SecondaryPrivateIpAddressCount: The number of secondary private IPv4 addresses to assign to a network interface. When you specify a number of secondary IPv4 addresses, Amazon EC2 selects these IP addresses within the subnet's IPv4 CIDR range. You can't specify this option and specify more than one private IP address using privateIpAddresses .
            The number of IP addresses you can assign to a network interface varies by instance type. For more information, see IP Addresses Per ENI Per Instance Type in the Amazon Virtual Private Cloud User Guide .
            

    :type Ipv6Addresses: list
    :param Ipv6Addresses: One or more specific IPv6 addresses from the IPv6 CIDR block range of your subnet. You can't use this option if you're specifying a number of IPv6 addresses.
            (dict) --Describes an IPv6 address.
            Ipv6Address (string) --The IPv6 address.
            
            

    :type Ipv6AddressCount: integer
    :param Ipv6AddressCount: The number of IPv6 addresses to assign to a network interface. Amazon EC2 automatically selects the IPv6 addresses from the subnet range. You can't use this option if specifying specific IPv6 addresses. If your subnet has the AssignIpv6AddressOnCreation attribute set to true , you can specify 0 to override this setting.

    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :rtype: dict
    :return: {
        'NetworkInterface': {
            'NetworkInterfaceId': 'string',
            'SubnetId': 'string',
            'VpcId': 'string',
            'AvailabilityZone': 'string',
            'Description': 'string',
            'OwnerId': 'string',
            'RequesterId': 'string',
            'RequesterManaged': True|False,
            'Status': 'available'|'attaching'|'in-use'|'detaching',
            'MacAddress': 'string',
            'PrivateIpAddress': 'string',
            'PrivateDnsName': 'string',
            'SourceDestCheck': True|False,
            'Groups': [
                {
                    'GroupName': 'string',
                    'GroupId': 'string'
                },
            ],
            'Attachment': {
                'AttachmentId': 'string',
                'InstanceId': 'string',
                'InstanceOwnerId': 'string',
                'DeviceIndex': 123,
                'Status': 'attaching'|'attached'|'detaching'|'detached',
                'AttachTime': datetime(2015, 1, 1),
                'DeleteOnTermination': True|False
            },
            'Association': {
                'PublicIp': 'string',
                'PublicDnsName': 'string',
                'IpOwnerId': 'string',
                'AllocationId': 'string',
                'AssociationId': 'string'
            },
            'TagSet': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'PrivateIpAddresses': [
                {
                    'PrivateIpAddress': 'string',
                    'PrivateDnsName': 'string',
                    'Primary': True|False,
                    'Association': {
                        'PublicIp': 'string',
                        'PublicDnsName': 'string',
                        'IpOwnerId': 'string',
                        'AllocationId': 'string',
                        'AssociationId': 'string'
                    }
                },
            ],
            'Ipv6Addresses': [
                {
                    'Ipv6Address': 'string'
                },
            ],
            'InterfaceType': 'interface'|'natGateway'
        }
    }
    
    
    """
    pass