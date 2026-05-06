def authorize_security_group_egress(DryRun=None, GroupId=None, SourceSecurityGroupName=None, SourceSecurityGroupOwnerId=None, IpProtocol=None, FromPort=None, ToPort=None, CidrIp=None, IpPermissions=None):
    """
    [EC2-VPC only] Adds one or more egress rules to a security group for use with a VPC. Specifically, this action permits instances to send traffic to one or more destination IPv4 or IPv6 CIDR address ranges, or to one or more destination security groups for the same VPC. This action doesn't apply to security groups for use in EC2-Classic. For more information, see Security Groups for Your VPC in the Amazon Virtual Private Cloud User Guide . For more information about security group limits, see Amazon VPC Limits .
    Each rule consists of the protocol (for example, TCP), plus either a CIDR range or a source group. For the TCP and UDP protocols, you must also specify the destination port or port range. For the ICMP protocol, you must also specify the ICMP type and code. You can use -1 for the type or code to mean all types or all codes.
    Rule changes are propagated to affected instances as quickly as possible. However, a small delay might occur.
    See also: AWS API Documentation
    
    
    :example: response = client.authorize_security_group_egress(
        DryRun=True|False,
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

    :type GroupId: string
    :param GroupId: [REQUIRED]
            The ID of the security group.
            

    :type SourceSecurityGroupName: string
    :param SourceSecurityGroupName: The name of a destination security group. To authorize outbound access to a destination security group, we recommend that you use a set of IP permissions instead.

    :type SourceSecurityGroupOwnerId: string
    :param SourceSecurityGroupOwnerId: The AWS account number for a destination security group. To authorize outbound access to a destination security group, we recommend that you use a set of IP permissions instead.

    :type IpProtocol: string
    :param IpProtocol: The IP protocol name or number. We recommend that you specify the protocol in a set of IP permissions instead.

    :type FromPort: integer
    :param FromPort: The start of port range for the TCP and UDP protocols, or an ICMP type number. We recommend that you specify the port range in a set of IP permissions instead.

    :type ToPort: integer
    :param ToPort: The end of port range for the TCP and UDP protocols, or an ICMP type number. We recommend that you specify the port range in a set of IP permissions instead.

    :type CidrIp: string
    :param CidrIp: The CIDR IPv4 address range. We recommend that you specify the CIDR range in a set of IP permissions instead.

    :type IpPermissions: list
    :param IpPermissions: A set of IP permissions. You can't specify a destination security group and a CIDR IP address range.
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