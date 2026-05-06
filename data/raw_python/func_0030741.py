def datasets(self):
        """
        Return all datasets

        :return:
        """

        return self.session.query(Dataset).filter(Dataset.vid != ROOT_CONFIG_NAME_V).all()