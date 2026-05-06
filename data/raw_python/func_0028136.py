def get_showcases(self):
        # type: () -> List[hdx.data.showcase.Showcase]
        """Get any showcases the dataset is in

        Returns:
            List[Showcase]: list of showcases
        """
        assoc_result, showcases_dicts = self._read_from_hdx('showcase', self.data['id'], fieldname='package_id',
                                                            action=hdx.data.showcase.Showcase.actions()['list_showcases'])
        showcases = list()
        if assoc_result:
            for showcase_dict in showcases_dicts:
                showcase = hdx.data.showcase.Showcase(showcase_dict, configuration=self.configuration)
                showcases.append(showcase)
        return showcases