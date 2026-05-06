def authorize_security_group_ingress(DryRun=None, GroupName=None, GroupId=None, SourceSecurityGroupName=None, SourceSecurityGroupOwnerId=None, IpProtocol=None, FromPort=None, ToPort=None, CidrIp=None, IpPermissions=None):
    """
    Adds one or more ingress rules to a security group.
    Rule changes are propagated to instances within the security group as quickly as possible. However, a small delay might occur.
    [EC2-Classic] This action gives one or more IPv4 CIDR address ranges permission to access a security group in your account, or gives one or more security groups (called the source groups ) permission to access a security group for your account. A source group can be for your own AWS account, or another. You can have up to 100 rules per group.
    [EC2-VPC] This action gives one or more IPv4 or IPv6 CIDR address ranges permission to access a security group in your VPC, or gives one or more other security groups (called the source groups ) permission to access a security group for your VPC. The security groups must all be for the same VPC or a peer VPC in a VPC peering connection. For more information about VPC security group limits, see Amazon VPC Limits .
    See also: AWS API Documentation
    
    
    :example: response = client.authorize_security_group_ingress(
        DryRun=True|False,
        GroupName='string',
        GroupId='string',
        SourceSecurityGroupName='string',
        SourceSecurityGroupOwnerId='string',
        IpProtocol='string',
        FromPort=123,
        ToPort=123,
        CidrIp='string',
        IpPermissions=[
            {
                'IpProtocol': 'string',
                'FromPort': 123,
                'ToPort': 123,
                'UserIdGroupPairs': [
                    {
                        'UserId': 'string',
                        'GroupName': 'string',
                        'GroupId': 'string',
                        'VpcId': 'string',
                        'VpcPeeringConnectionId': 'string',
                        'PeeringStatus': 'string'
                    },
                ],
                'IpRanges': [
                    {
                        'CidrIp': 'string'
                    },
                ],
                'Ipv6Ranges': [
                    {
                        'CidrIpv6': 'string'
                    },
                ],
                'PrefixListIds': [
                    {
                        'PrefixListId': 'string'
                    },
                ]
            },
        ]
    )
    
    
    :type DryRun: boolean
    :param DryRun: Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .

    :type GroupName: string
    :param GroupName: [EC2-Classic, default VPC] The name of the security group.

    :type GroupId: string
    :param GroupId: The ID of the security group. Required for a nondefault VPC.

    :type SourceSecurityGroupName: string
    :param SourceSecurityGroupName: [EC2-Classic, default VPC] The name of the source security group. You can't specify this parameter in combination with the following parameters: the CIDR IP address range, the start of the port range, the IP protocol, and the end of the port range. Creates rules that grant full ICMP, UDP, and TCP access. To create a rule with a specific IP protocol and port range, use a set of IP permissions instead. For EC2-VPC, the source security group must be in the same VPC.

    :type SourceSecurityGroupOwnerId: string
    :param SourceSecurityGroupOwnerId: [EC2-Classic] The AWS account number for the source security group, if the source security group is in a different account. You can't specify this parameter in combination with the following parameters: the CIDR IP address range, the IP protocol, the start of the port range, and the end of the port range. Creates rules that grant full ICMP, UDP, and TCP access. To create a rule with a specific IP protocol and port range, use a set of IP permissions instead.

    :type IpProtocol: string
    :param IpProtocol: The IP protocol name (tcp , udp , icmp ) or number (see Protocol Numbers ). (VPC only) Use -1 to specify all protocols. If you specify -1 , or a protocol number other than tcp , udp , icmp , or 58 (ICMPv6), traffic on all ports is allowed, regardless of any ports you specify. For tcp , udp , and icmp , you must specify a port range. For protocol 58 (ICMPv6), you can optionally specify a port range; if you don't, traffic for all types and codes is allowed.

    :type FromPort: integer
    :param FromPort: The start of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 type number. For the ICMP/ICMPv6 type number, use -1 to specify all types.

    :type ToPort: integer
    :param ToPort: The end of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 code number. For the ICMP/ICMPv6 code number, use -1 to specify all codes.

    :type CidrIp: string
    :param CidrIp: The CIDR IPv4 address range. You can't specify this parameter when specifying a source security group.

    :type IpPermissions: list
    :param IpPermissions: A set of IP permissions. Can be used to specify multiple rules in a single command.
            (dict) --Describes a security group rule.
            IpProtocol (string) --The IP protocol name (tcp , udp , icmp ) or number (see Protocol Numbers ).
            [EC2-VPC only] Use -1 to specify all protocols. When authorizing security group rules, specifying -1 or a protocol number other than tcp , udp , icmp , or 58 (ICMPv6) allows traffic on all ports, regardless of any port range you specify. For tcp , udp , and icmp , you must specify a port range. For 58 (ICMPv6), you can optionally specify a port range; if you don't, traffic for all types and codes is allowed when authorizing rules.
            FromPort (integer) --The start of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 type number. A value of -1 indicates all ICMP/ICMPv6 types.
            ToPort (integer) --The end of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 code. A value of -1 indicates all ICMP/ICMPv6 codes for the specified ICMP type.
            UserIdGroupPairs (list) --One or more security group and AWS account ID pairs.
            (dict) --Describes a security group and AWS account ID pair.
            UserId (string) --The ID of an AWS account. For a referenced security group in another VPC, the account ID of the referenced security group is returned.
            [EC2-Classic] Required when adding or removing rules that reference a security group in another AWS account.
            GroupName (string) --The name of the security group. In a request, use this parameter for a security group in EC2-Classic or a default VPC only. For a security group in a nondefault VPC, use the security group ID.
            GroupId (string) --The ID of the security group.
            VpcId (string) --The ID of the VPC for the referenced security group, if applicable.
            VpcPeeringConnectionId (string) --The ID of the VPC peering connection, if applicable.
            PeeringStatus (string) --The status of a VPC peering connection, if applicable.
            
            IpRanges (list) --One or more IPv4 ranges.
            (dict) --Describes an IPv4 range.
            CidrIp (string) --The IPv4 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv4 address, use the /32 prefix.
            
            Ipv6Ranges (list) --[EC2-VPC only] One or more IPv6 ranges.
            (dict) --[EC2-VPC only] Describes an IPv6 range.
            CidrIpv6 (string) --The IPv6 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv6 address, use the /128 prefix.
            
            PrefixListIds (list) --(Valid for AuthorizeSecurityGroupEgress , RevokeSecurityGroupEgress and DescribeSecurityGroups only) One or more prefix list IDs for an AWS service. In an AuthorizeSecurityGroupEgress request, this is the AWS service that you want to access through a VPC endpoint from instances associated with the security group.
            (dict) --The ID of the prefix.
            PrefixListId (string) --The ID of the prefix.
            
            
            

    """
    pass