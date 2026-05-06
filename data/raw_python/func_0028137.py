def _get_dataset_showcase_dict(self, showcase):
        # type: (Union[hdx.data.showcase.Showcase, Dict,str]) -> Dict
        """Get dataset showcase dict

        Args:
            showcase (Union[Showcase,Dict,str]): Either a showcase id or Showcase metadata from a Showcase object or dictionary

        Returns:
            dict: dataset showcase dict
        """
        if isinstance(showcase, hdx.data.showcase.Showcase) or isinstance(showcase, dict):
            if 'id' not in showcase:
                showcase = hdx.data.showcase.Showcase.read_from_hdx(showcase['name'])
            showcase = showcase['id']
        elif not isinstance(showcase, str):
            raise HDXError('Type %s cannot be added as a showcase!' % type(showcase).__name__)
        if is_valid_uuid(showcase) is False:
            raise HDXError('%s is not a valid showcase id!' % showcase)
        return {'package_id': self.data['id'], 'showcase_id': showcase}