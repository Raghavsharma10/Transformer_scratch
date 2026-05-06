def _is_ready(self, topic_name):
        '''
        Is NSQ running and have space to receive messages?
        '''
        url = 'http://%s/stats?format=json&topic=%s' % (self.nsqd_http_address, topic_name)
        #Cheacking for ephmeral channels
        if '#' in topic_name:
            topic_name, tag =topic_name.split("#", 1)

        try:
            data = self.session.get(url).json()
            '''
            data = {u'start_time': 1516164866, u'version': u'1.0.0-compat', \
                    u'health': u'OK', u'topics': [{u'message_count': 19019, \
                    u'paused': False, u'topic_name': u'test_topic', u'channels': [], \
                    u'depth': 19019, u'backend_depth': 9019, u'e2e_processing_latency': {u'count': 0, \
                    u'percentiles': None}}]}
            '''
            topics = data.get('topics', [])
            topics = [t for t in topics if t['topic_name'] == topic_name]

            if not topics:
                raise Exception('topic_missing_at_nsq')

            topic = topics[0]
            depth = topic['depth']
            depth += sum(c.get('depth', 0) for c in topic['channels'])
            self.log.debug('nsq_depth_check', topic=topic_name,
                            depth=depth, max_depth=self.nsq_max_depth)

            if depth < self.nsq_max_depth:
                return
            else:
                raise Exception('nsq_is_full_waiting_to_clear')
        except:
            raise