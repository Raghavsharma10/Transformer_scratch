def _add_config_root(self):
        """ Adds the root dataset, which holds configuration values for the database. """

        try:
            self.session.query(Dataset).filter_by(id=ROOT_CONFIG_NAME).one()
            self.close_session()
        except NoResultFound:
            o = Dataset(
                id=ROOT_CONFIG_NAME,
                vid=ROOT_CONFIG_NAME_V,
                name=ROOT_CONFIG_NAME,
                vname=ROOT_CONFIG_NAME_V,
                fqname='datasetroot-0.0.0~' + ROOT_CONFIG_NAME_V,
                cache_key=ROOT_CONFIG_NAME,
                version='0.0.0',
                source=ROOT_CONFIG_NAME,
                dataset=ROOT_CONFIG_NAME,
                revision=1,
            )
            self.session.add(o)
            self.commit()