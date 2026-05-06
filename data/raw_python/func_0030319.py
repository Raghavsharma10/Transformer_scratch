def _put_metadata(self, fs_remote, ds):
        """Store metadata on a pyfs remote"""

        from six import text_type
        from fs.errors import ResourceNotFoundError


        identity = ds.identity
        d = identity.dict

        d['summary'] = ds.config.metadata.about.summary
        d['title'] = ds.config.metadata.about.title

        meta_stack = self._meta_infos(ds)

        def do_metadata():
            for path, ident in meta_stack:
                fs_remote.setcontents(path, ident)


        try:
            # Assume the directories already exist
            do_metadata()
        except ResourceNotFoundError:
            # Nope, make them and try again.
            parts = ['vid', 'id', 'vname', 'name']
            for p in parts:
                dirname = os.path.join('_meta', p)
                fs_remote.makedir(dirname, allow_recreate=True, recursive=True)

            do_metadata()