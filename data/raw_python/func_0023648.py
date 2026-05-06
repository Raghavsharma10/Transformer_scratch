def clean_stats(self):
        '''Stats with topics and channels keyed on topic and channel names'''
        stats = self.stats()
        if 'topics' in stats:  # pragma: no branch
            topics = stats['topics']
            topics = dict((t.pop('topic_name'), t) for t in topics)
            for topic, data in topics.items():
                if 'channels' in data:  # pragma: no branch
                    channels = data['channels']
                    channels = dict(
                        (c.pop('channel_name'), c) for c in channels)
                    data['channels'] = channels
            stats['topics'] = topics
        return stats