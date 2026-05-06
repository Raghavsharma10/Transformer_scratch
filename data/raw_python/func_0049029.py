def build_statistics_item(self):
        """
        Statistics items are as following:
        * zabbix_sender result
            + processed, failed, total and more...
        """
        if self.result is not None:
            stats = dict()
            result = self.get_result()
            info = result['info']
            info = info.split(';')
            info = [entry.split(':') for entry in info]

            prefix = 'blackbird.zabbix_sender'

            for entry in info:
                key = entry[0].strip()
                value = None

                if key == 'processed':
                    value = int(entry[1])
                elif key == 'failed':
                    value = int(entry[1])
                elif key == 'total':
                    value = int(entry[1])
                elif key == 'seconds spent':
                    key = key.replace(' ', '_')
                    value = float(entry[1])
                    value *= 1000
                    value = str(round(value, 6))
                else:
                    log_message = (
                        'Blackbird has never seen {key}. {key} is new key??'
                        ''.format(key=key)
                    )
                    self.logger.info(log_message)

                if value is not None:
                    key = '.'.join([prefix, key])
                    stats[key] = value

            if 'response' in result:
                key = '.'.join([prefix, 'response'])
                stats[key] = result['response']

            for key, value in stats.iteritems():
                stats_key_list = [
                    'blackbird.zabbix_sender.processed',
                    'blackbird.zabbix_sender.failed',
                    'blackbird.zabbix_sender.total',
                ]
                item = BlackbirdStatisticsItem(
                    key=key,
                    value=value,
                    host=self.options['hostname']
                )
                if key in stats_key_list:
                    if self.enqueue(item=item, queue=self.stats_queue):
                        self.logger.debug(
                            'Inserted {0} to the statistics queue'
                            ''.format(item.data)
                        )
                else:
                    if self.enqueue(item=item, queue=self.queue):
                        self.logger.debug(
                            'Inserted {0} to the queue'.format(item.data)
                        )