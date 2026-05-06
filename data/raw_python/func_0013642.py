def ping(self, destination, source=c.PING_SOURCE, ttl=c.PING_TTL, timeout=c.PING_TIMEOUT,
             size=c.PING_SIZE, count=c.PING_COUNT, vrf=c.PING_VRF):
        """
        Executes ping on the device and returns a dictionary with the result

        :param destination: Host or IP Address of the destination
        :param source (optional): Source address of echo request
        :param ttl (optional): Maximum number of hops
        :param timeout (optional): Maximum seconds to wait after sending final packet
        :param size (optional): Size of request (bytes)
        :param count (optional): Number of ping request to send

        Output dictionary has one of following keys:

            * success
            * error

        In case of success, inner dictionary will have the followin keys:

            * probes_sent (int)
            * packet_loss (int)
            * rtt_min (float)
            * rtt_max (float)
            * rtt_avg (float)
            * rtt_stddev (float)
            * results (list)

        'results' is a list of dictionaries with the following keys:

            * ip_address (str)
            * rtt (float)

        Example::

            {
                'success': {
                    'probes_sent': 5,
                    'packet_loss': 0,
                    'rtt_min': 72.158,
                    'rtt_max': 72.433,
                    'rtt_avg': 72.268,
                    'rtt_stddev': 0.094,
                    'results': [
                        {
                            'ip_address': u'1.1.1.1',
                            'rtt': 72.248
                        },
                        {
                            'ip_address': '2.2.2.2',
                            'rtt': 72.299
                        }
                    ]
                }
            }

            OR

            {
                'error': 'unknown host 8.8.8.8.8'
            }

        """
        raise NotImplementedError