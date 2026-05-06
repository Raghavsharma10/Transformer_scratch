def parse_env_zones(self):
        '''returns a list of comma separated zones parsed from the GCE_ZONE environment variable.
        If provided, this will be used to filter the results of the grouped_instances call'''
        import csv
        reader = csv.reader([os.environ.get('GCE_ZONE',"")], skipinitialspace=True)
        zones = [r for r in reader]
        return [z for z in zones[0]]