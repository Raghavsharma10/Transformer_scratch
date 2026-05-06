def save_index(self, filename):
        ''' Save the current Layout's index to a .json file.

        Args:
            filename (str): Filename to write to.

        Note: At the moment, this won't serialize directory-specific config
        files. This means reconstructed indexes will only work properly in
        cases where there aren't multiple layout specs within a project.
        '''
        data = {}
        for f in self.files.values():
            entities = {v.entity.id: v.value for k, v in f.tags.items()}
            data[f.path] = {'domains': f.domains, 'entities': entities}
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)