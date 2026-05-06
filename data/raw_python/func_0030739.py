def package_dataset(self):
        """For sqlite bundle packages, return the first ( and only ) dataset"""

        return self.session.query(Dataset).filter(Dataset.vid != ROOT_CONFIG_NAME_V).one()