def new_from_bundle_config(self, config):
        """
        Create a new bundle, or link to an existing one, based on the identity in config data.

        :param config: A Dict form of a bundle.yaml file
        :return:
        """
        identity = Identity.from_dict(config['identity'])

        ds = self._db.dataset(identity.vid, exception=False)

        if not ds:
            ds = self._db.new_dataset(**identity.dict)

        b = Bundle(ds, self)
        b.commit()
        b.state = Bundle.STATES.NEW
        b.set_last_access(Bundle.STATES.NEW)

        # b.set_file_system(source_url=self._fs.source(ds.name),
        #                   build_url=self._fs.build(ds.name))

        return b