def add_showcase(self, showcase, showcases_to_check=None):
        # type: (Union[hdx.data.showcase.Showcase,Dict,str], List[hdx.data.showcase.Showcase]) -> bool
        """Add dataset to showcase

        Args:
            showcase (Union[Showcase,Dict,str]): Either a showcase id or showcase metadata from a Showcase object or dictionary
            showcases_to_check (List[Showcase]): list of showcases against which to check existence of showcase. Defaults to showcases containing dataset.

        Returns:
            bool: True if the showcase was added, False if already present
        """
        dataset_showcase = self._get_dataset_showcase_dict(showcase)
        if showcases_to_check is None:
            showcases_to_check = self.get_showcases()
        for showcase in showcases_to_check:
            if dataset_showcase['showcase_id'] == showcase['id']:
                return False
        showcase = hdx.data.showcase.Showcase({'id': dataset_showcase['showcase_id']}, configuration=self.configuration)
        showcase._write_to_hdx('associate', dataset_showcase, 'package_id')
        return True