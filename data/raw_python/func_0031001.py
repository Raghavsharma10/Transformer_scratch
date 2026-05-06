def load_index(self, filename, reindex=False):
        ''' Load the Layout's index from a plaintext file.

        Args:
            filename (str): Path to the plaintext index file.
            reindex (bool): If True, discards entity values provided in the
                loaded index and instead re-indexes every file in the loaded
                index against the entities defined in the config. Default is
                False, in which case it is assumed that all entity definitions
                in the loaded index are correct and do not need any further
                validation.

        Note: At the moment, directory-specific config files aren't serialized.
        This means reconstructed indexes will only work properly in cases
        where there aren't multiple layout specs within a project.
        '''
        self._reset_index()
        with open(filename, 'r') as fobj:
            data = json.load(fobj)

        for path, file in data.items():

            ents, domains = file['entities'], file['domains']

            root, f = dirname(path), basename(path)
            if reindex:
                self._index_file(root, f, domains)
            else:
                f = self._make_file_object(root, f)
                tags = {k: Tag(self.entities[k], v) for k, v in ents.items()}
                f.tags = tags
                self.files[f.path] = f

                for ent, val in f.entities.items():
                    self.entities[ent].add_file(f.path, val)