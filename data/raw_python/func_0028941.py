def _get_notification_spec(self, lambda_arn):
        lambda_name = base.get_lambda_name(lambda_arn)
        notification_spec = {
            'Id': self._make_notification_id(lambda_name),
            'Events': [e for e in self._config['events']],
            'LambdaFunctionArn': lambda_arn
        }

        # Add S3 key filters
        filter_rules = []
        # look for filter rules
        for filter_type in ['prefix', 'suffix']:
            if filter_type in self._config:
                rule = {'Name': filter_type.capitalize(), 'Value': self._config[filter_type] }
                filter_rules.append(rule)

        if filter_rules:
            notification_spec['Filter'] = {'Key': {'FilterRules': filter_rules } }
        '''
        if 'key_filters' in self._config:
            filters_spec = {'Key': {'FilterRules': [] } }
            # I do not think this is a useful structure:
            for filter in self._config['key_filters']:
                if 'type' in filter and 'value' in filter and filter['type'] in ('prefix', 'suffix'):
                    rule = {'Name': filter['type'].capitalize(), 'Value': filter['value'] }
                    filters_spec['Key']['FilterRules'].append(rule)

            notification_spec['Filter'] = filters_spec
        '''
        return notification_spec