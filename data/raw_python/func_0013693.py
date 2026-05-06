def create_network_profile(projectArn=None, name=None, description=None, type=None, uplinkBandwidthBits=None, downlinkBandwidthBits=None, uplinkDelayMs=None, downlinkDelayMs=None, uplinkJitterMs=None, downlinkJitterMs=None, uplinkLossPercent=None, downlinkLossPercent=None):
    """
    Creates a network profile.
    See also: AWS API Documentation
    
    
    :example: response = client.create_network_profile(
        projectArn='string',
        name='string',
        description='string',
        type='CURATED'|'PRIVATE',
        uplinkBandwidthBits=123,
        downlinkBandwidthBits=123,
        uplinkDelayMs=123,
        downlinkDelayMs=123,
        uplinkJitterMs=123,
        downlinkJitterMs=123,
        uplinkLossPercent=123,
        downlinkLossPercent=123
    )
    
    
    :type projectArn: string
    :param projectArn: [REQUIRED]
            The Amazon Resource Name (ARN) of the project for which you want to create a network profile.
            

    :type name: string
    :param name: [REQUIRED]
            The name you wish to specify for the new network profile.
            

    :type description: string
    :param description: The description of the network profile.

    :type type: string
    :param type: The type of network profile you wish to create. Valid values are listed below.

    :type uplinkBandwidthBits: integer
    :param uplinkBandwidthBits: The data throughput rate in bits per second, as an integer from 0 to 104857600.

    :type downlinkBandwidthBits: integer
    :param downlinkBandwidthBits: The data throughput rate in bits per second, as an integer from 0 to 104857600.

    :type uplinkDelayMs: integer
    :param uplinkDelayMs: Delay time for all packets to destination in milliseconds as an integer from 0 to 2000.

    :type downlinkDelayMs: integer
    :param downlinkDelayMs: Delay time for all packets to destination in milliseconds as an integer from 0 to 2000.

    :type uplinkJitterMs: integer
    :param uplinkJitterMs: Time variation in the delay of received packets in milliseconds as an integer from 0 to 2000.

    :type downlinkJitterMs: integer
    :param downlinkJitterMs: Time variation in the delay of received packets in milliseconds as an integer from 0 to 2000.

    :type uplinkLossPercent: integer
    :param uplinkLossPercent: Proportion of transmitted packets that fail to arrive from 0 to 100 percent.

    :type downlinkLossPercent: integer
    :param downlinkLossPercent: Proportion of received packets that fail to arrive from 0 to 100 percent.

    :rtype: dict
    :return: {
        'networkProfile': {
            'arn': 'string',
            'name': 'string',
            'description': 'string',
            'type': 'CURATED'|'PRIVATE',
            'uplinkBandwidthBits': 123,
            'downlinkBandwidthBits': 123,
            'uplinkDelayMs': 123,
            'downlinkDelayMs': 123,
            'uplinkJitterMs': 123,
            'downlinkJitterMs': 123,
            'uplinkLossPercent': 123,
            'downlinkLossPercent': 123
        }
    }
    
    
    """
    pass