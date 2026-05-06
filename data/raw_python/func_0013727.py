def replace_network_acl_entry(DryRun=None, NetworkAclId=None, RuleNumber=None, Protocol=None, RuleAction=None, Egress=None, CidrBlock=None, Ipv6CidrBlock=None, IcmpTypeCode=None, PortRange=None):
    """
    Replaces an entry (rule) in a network ACL. For more information about network ACLs, see Network ACLs in the Amazon Virtual Private Cloud User Guide .
    See also: AWS API Documentation
    
    Examples
    This example replaces an entry for the specified network ACL. The new rule 100 allows ingress traffic from 203.0.113.12/24 on UDP port 53 (DNS) into any associated subnet.
    Expected Output:
    
    :example: response = client.replace_network_acl_entry(
        DryRun=True|False,
        NetworkAclId='string',
        RuleNumber=123,
        Protocol='string',
        RuleAction='allow'|'deny',
        Egress=True|False,
        CidrBlock='string',
        Ipv6CidrBlock='string',
        IcmpTypeCode={
            'Type': 123,
            'Code': 123
        },
        PortRange={
            'From': 123,
            'To': 123
        }
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type NetworkAclId: string
    :param NetworkAclId: [REQUIRED]
            The ID of the ACL.
            

    :type RuleNumber: integer
    :param RuleNumber: [REQUIRED]
            The rule number of the entry to replace.
            

    :type Protocol: string
    :param Protocol: [REQUIRED]
            The IP protocol. You can specify all or -1 to mean all protocols. If you specify all , -1 , or a protocol number other than tcp , udp , or icmp , traffic on all ports is allowed, regardless of any ports or ICMP types or codes you specify. If you specify protocol 58 (ICMPv6) and specify an IPv4 CIDR block, traffic for all ICMP types and codes allowed, regardless of any that you specify. If you specify protocol 58 (ICMPv6) and specify an IPv6 CIDR block, you must specify an ICMP type and code.
            

    :type RuleAction: string
    :param RuleAction: [REQUIRED]
            Indicates whether to allow or deny the traffic that matches the rule.
            

    :type Egress: boolean
    :param Egress: [REQUIRED]
            Indicates whether to replace the egress rule.
            Default: If no value is specified, we replace the ingress rule.
            

    :type CidrBlock: string
    :param CidrBlock: The IPv4 network range to allow or deny, in CIDR notation (for example 172.16.0.0/24 ).

    :type Ipv6CidrBlock: string
    :param Ipv6CidrBlock: The IPv6 network range to allow or deny, in CIDR notation (for example 2001:bd8:1234:1a00::/64 ).

    :type IcmpTypeCode: dict
    :param IcmpTypeCode: ICMP protocol: The ICMP or ICMPv6 type and code. Required if specifying the ICMP (1) protocol, or protocol 58 (ICMPv6) with an IPv6 CIDR block.
            Type (integer) --The ICMP type. A value of -1 means all types.
            Code (integer) --The ICMP code. A value of -1 means all codes for the specified ICMP type.
            

    :type PortRange: dict
    :param PortRange: TCP or UDP protocols: The range of ports the rule applies to. Required if specifying TCP (6) or UDP (17) for the protocol.
            From (integer) --The first port in the range.
            To (integer) --The last port in the range.
            

    :return: response = client.replace_network_acl_entry(
        CidrBlock='203.0.113.12/24',
        Egress=False,
        NetworkAclId='acl-5fb85d36',
        PortRange={
            'From': 53,
            'To': 53,
        },
        Protocol='udp',
        RuleAction='allow',
        RuleNumber=100,
    )
    
    print(response)
    
    
    """
    pass