def from_filename(self, filename):
        '''
        Build an IntentSchema from a file path 
        creates a new intent schema if the file does not exist, throws an error if the file
        exists but cannot be loaded as a JSON
        '''
        if os.path.exists(filename):
            with open(filename) as fp:
                return IntentSchema(json.load(fp, object_pairs_hook=OrderedDict))
        else:
            print ('File does not exist')
            return IntentSchema()