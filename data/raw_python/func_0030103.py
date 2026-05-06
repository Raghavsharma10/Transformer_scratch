def new_bundle(self, assignment_class=None, **kwargs):
        """
        Create a new bundle, with the same arguments as creating a new dataset

        :param assignment_class: String. assignment class to use for fetching a number, if one
        is not specified in kwargs
        :param kwargs:
        :return:
        """

        if not ('id' in kwargs and bool(kwargs['id'])) or assignment_class is not None:
            kwargs['id'] = self.number(assignment_class)

        ds = self._db.new_dataset(**kwargs)
        self._db.commit()

        b = self.bundle(ds.vid)
        b.state = Bundle.STATES.NEW

        b.set_last_access(Bundle.STATES.NEW)

        b.set_file_system(source_url=self._fs.source(b.identity.source_path),
                          build_url=self._fs.build(b.identity.source_path))

        bs_meta = b.build_source_files.file(File.BSFILE.META)
        bs_meta.set_defaults()
        bs_meta.record_to_objects()
        bs_meta.objects_to_record()
        b.commit()

        self._db.commit()
        return b