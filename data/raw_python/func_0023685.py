def raw(self):
        '''All the raw, unaggregated stats (with duplicates).'''
        topic_keys = (
            'message_count',
            'depth',
            'backend_depth',
            'paused'
        )

        channel_keys = (
            'in_flight_count',
            'timeout_count',
            'paused',
            'deferred_count',
            'message_count',
            'depth',
            'backend_depth',
            'requeue_count'
        )
        
        for host, stats in self.merged.items():
            for topic, stats in stats.get('topics', {}).items():
                prefix = 'host.%s.topic.%s' % (host, topic)
                for key in topic_keys:
                    value = int(stats.get(key, -1))
                    yield (
                        'host.%s.topic.%s.%s' % (host, topic, key),
                        value,
                        False
                    )
                    yield (
                        'topic.%s.%s' % (topic, key),
                        value,
                        True
                    )
                    yield (
                        'topics.%s' % key,
                        value,
                        True
                    )
                
                for chan, stats in stats.get('channels', {}).items():
                    data = {
                        key: int(stats.get(key, -1)) for key in channel_keys
                    }
                    data['clients'] = len(stats.get('clients', []))

                    for key, value in data.items():
                        yield (
                            'host.%s.topic.%s.channel.%s.%s' % (host, topic, chan, key),
                            value,
                            False
                        )
                        yield (
                            'host.%s.topic.%s.channels.%s' % (host, topic, key),
                            value,
                            True
                        )
                        yield (
                            'topic.%s.channels.%s' % (topic, key),
                            value,
                            True
                        )
                        yield (
                            'channels.%s' % key,
                            value,
                            True
                        )