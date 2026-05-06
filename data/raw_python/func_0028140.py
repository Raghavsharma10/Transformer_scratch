def remove_showcase(self, showcase):
        # type: (Union[hdx.data.showcase.Showcase,Dict,str]) -> None
        """Remove dataset from showcase

        Args:
            showcase (Union[Showcase,Dict,str]): Either a showcase id string or showcase metadata from a Showcase object or dictionary

        Returns:
            None
        """
        dataset_showcase = self._get_dataset_showcase_dict(showcase)
        showcase = hdx.data.showcase.Showcase({'id': dataset_showcase['showcase_id']}, configuration=self.configuration)
        showcase._write_to_hdx('disassociate', dataset_showcase, 'package_id')