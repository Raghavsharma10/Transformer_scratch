def update_network_profile(arn=None, name=None, description=None, type=None, uplinkBandwidthBits=None, downlinkBandwidthBits=None, uplinkDelayMs=None, downlinkDelayMs=None, uplinkJitterMs=None, downlinkJitterMs=None, uplinkLossPercent=None, downlinkLossPercent=None):
    """
    Updates the network profile with specific settings.
    See also: AWS API Documentation
    
    
    :example: response = client.update_network_profile(
        arn='string',
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
    
    
    :type arn: string
    :param arn: [REQUIRED]
            The Amazon Resource Name (ARN) of the project that you wish to update network profile settings.
            

    :type name: string
    :param name: The name of the network profile about which you are returning information.

    :type description: string
    :param description: The descriptoin of the network profile about which you are returning information.

    :type type: string
    :param type: The type of network profile you wish to return information about. Valid values are listed below.

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