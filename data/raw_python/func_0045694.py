def retrieve(self, key_term):
        """Return data for key term specified for each resolved name as a list.
Possible terms (02/12/2013): 'query_name', 'classification_path',
'data_source_title', 'match_type', 'score', 'classification_path_ranks',
'name_string', 'canonical_form',\
'classification_path_ids', 'prescore', 'data_source_id', 'taxon_id',
'gni_uuid'"""
        if key_term not in self.key_terms:
            raise IndexError('Term given is invalid! Check doc string for \
valid terms.')
        store = self._store
        retrieved = []
        for key in list(store.keys()):
            # take copy, so changes made to the returned list do not affect
            #  store
            record = copy.deepcopy(store[key])
            if len(record) > 0:
                if key_term == 'query_name':
                    retrieved.append(key)
                else:
                    retrieved.append(record[0][key_term])
        if re.search('path', key_term):
            retrieved = [[r2 for r2 in r1.split('|')[1:]] for r1 in retrieved]
        return retrieved