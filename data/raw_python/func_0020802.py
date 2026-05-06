def _create_das_mapping(self):
        """
        das_map = {'lookup' : [{params : {'param1' : 'required', 'param2' : 'optional', 'param3' : 'default_value' ...},
                                url : 'https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/acquisitioneras/',
                                das_map : {'das_param1' : dbs_param1, ...}
                                }]
                                }
        """
        with open(self._mapfile, 'r') as f:
            for entry in yaml.load_all(f):
                das2dbs_param_map = {}
                if 'lookup' not in entry:
                    continue
                for param_map in entry['das_map']:
                    if 'api_arg' in param_map:
                        das2dbs_param_map[param_map['das_key']] = param_map['api_arg']

                self._das_map.setdefault(entry['lookup'], []).append({'params' : entry['params'],
                                                                     'url' : entry['url'],
                                                                     'das2dbs_param_map' : das2dbs_param_map})