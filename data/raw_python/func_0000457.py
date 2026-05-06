def list_instances_json(self, application=None, show_only_destroyed=False):
        """ Get list of instances in json format converted to list"""
        # todo: application should not be parameter here. Application should do its own list, just in sake of code reuse
        q_filter = {'sortBy': 'byCreation', 'descending': 'true',
                    'mode': 'short',
                    'from': '0', 'to': '10000'}
        if not show_only_destroyed:
            q_filter['showDestroyed'] = 'false'
        else:
            q_filter['showDestroyed'] = 'true'
            q_filter['showRunning'] = 'false'
            q_filter['showError'] = 'false'
            q_filter['showLaunching'] = 'false'
        if application:
            q_filter["applicationFilterId"] = application.applicationId
        resp_json = self._router.get_instances(org_id=self.organizationId, params=q_filter).json()
        if type(resp_json) == dict:
            instances = [instance for g in resp_json['groups'] for instance in g['records']]
        else:  # TODO: This is compatibility fix for platform < 37.1
            instances = resp_json

        return instances