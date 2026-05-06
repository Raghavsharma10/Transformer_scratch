def update(self, data):
        """update known data with with newly provided data"""
        if not isinstance(data, list): data = [data] # otherwise no conversion is necessary
        master = Handler.ALL_VERS_DATA
        for record in data:
            #print(record)
            for k,v in iteritems(record): # ensure record contents aretyped appropriately
                try:                record[k] = int(v)
                except ValueError:  record[k] = v
            try: label = record["label"] # verify this record has the required 'label' key
            except KeyError:
                raise ValueError("Must provide a valid label argument.  Given:%s%s"%(\
                    os.linesep, ("%s  "%(os.linesep)).join(
                        ["%15s:%s"%(k,v) for k,v in iteritems(kwargs)]
                    )))
            try:    masterLabel = master[label] # identify the already existing record that matches this to-be-updated record, if any
            except KeyError: # master hasn't been defined yet
                master[label] = record
                self._updated = True # a new record should also be saved
                continue
            for k,v in iteritems(record): # determine whether master needs to be updated
                try:
                    if masterLabel[k] == v:  continue # whether an entry in the record needs to be updated (doesn't match)
                except KeyError:             pass # this condition means that k is a new key, so the record must be updated
                self._updated = True
                try:    master[label].update(record) # index each record by its label
                except KeyError:             break