def _update_fs_list(self):
        """Cache the full list for http access. This creates a meta file that can be read all at once,
        rather than requiring a list operation like S3 access does"""
        from json import dumps

        full_list = [ e[1] for e in self._list_fs(full=True) ]

        remote = self._fs_remote(self.url)

        remote.setcontents(os.path.join('_meta', 'list.json'), dumps(full_list, indent = 4))