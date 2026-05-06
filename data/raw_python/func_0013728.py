def replace_route(DryRun=None, RouteTableId=None, DestinationCidrBlock=None, GatewayId=None, DestinationIpv6CidrBlock=None, EgressOnlyInternetGatewayId=None, InstanceId=None, NetworkInterfaceId=None, VpcPeeringConnectionId=None, NatGatewayId=None):
    """
    Replaces an existing route within a route table in a VPC. You must provide only one of the following: Internet gateway or virtual private gateway, NAT instance, NAT gateway, VPC peering connection, network interface, or egress-only Internet gateway.
    For more information about route tables, see Route Tables in the Amazon Virtual Private Cloud User Guide .
    See also: AWS API Documentation
    
    Examples
    This example replaces the specified route in the specified table table. The new route matches the specified CIDR and sends the traffic to the specified virtual private gateway.
    Expected Output:
    
    :example: response = client.replace_route(
        DryRun=True|False,
        RouteTableId='string',
        DestinationCidrBlock='string',
        GatewayId='string',
        DestinationIpv6CidrBlock='string',
        EgressOnlyInternetGatewayId='string',
        InstanceId='string',
        NetworkInterfaceId='string',
        VpcPeeringConnectionId='string',
        NatGatewayId='string'
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type RouteTableId: string
    :param RouteTableId: [REQUIRED]
            The ID of the route table.
            

    :type DestinationCidrBlock: string
    :param DestinationCidrBlock: The IPv4 CIDR address block used for the destination match. The value you provide must match the CIDR of an existing route in the table.

    :type GatewayId: string
    :param GatewayId: The ID of an Internet gateway or virtual private gateway.

    :type DestinationIpv6CidrBlock: string
    :param DestinationIpv6CidrBlock: The IPv6 CIDR address block used for the destination match. The value you provide must match the CIDR of an existing route in the table.

    :type EgressOnlyInternetGatewayId: string
    :param EgressOnlyInternetGatewayId: [IPv6 traffic only] The ID of an egress-only Internet gateway.

    :type InstanceId: string
    :param InstanceId: The ID of a NAT instance in your VPC.

    :type NetworkInterfaceId: string
    :param NetworkInterfaceId: The ID of a network interface.

    :type VpcPeeringConnectionId: string
    :param VpcPeeringConnectionId: The ID of a VPC peering connection.

    :type NatGatewayId: string
    :param NatGatewayId: [IPv4 traffic only] The ID of a NAT gateway.

    :return: response = client.replace_route(
        DestinationCidrBlock='10.0.0.0/16',
        GatewayId='vgw-9a4cacf3',
        RouteTableId='rtb-22574640',
    )
    
    print(response)
    
    
    """
    pass