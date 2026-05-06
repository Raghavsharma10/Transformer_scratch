def create_subscriptions(config, profile_name):
    ''' Adds supported subscriptions '''
    if 'kinesis' in config.subscription.keys():
        data = config.subscription['kinesis']
        function_name = config.name
        stream_name = data['stream']
        batch_size = data['batch_size']
        starting_position = data['starting_position']
        starting_position_ts = None
        if starting_position == 'AT_TIMESTAMP':
            ts = data.get('starting_position_timestamp')
            starting_position_ts = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
        s = KinesisSubscriber(config, profile_name,
                              function_name, stream_name, batch_size,
                              starting_position,
                              starting_position_ts=starting_position_ts)
        s.subscribe()