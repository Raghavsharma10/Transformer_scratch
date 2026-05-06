def _get_showcase_dataset_dict(self, dataset):
        # type: (Union[hdx.data.dataset.Dataset,Dict,str]) -> Dict
        """Get showcase dataset dict

        Args:
            showcase (Union[Showcase,Dict,str]): Either a showcase id or Showcase metadata from a Showcase object or dictionary

        Returns:
            Dict: showcase dataset dict
        """
        if isinstance(dataset, hdx.data.dataset.Dataset) or isinstance(dataset, dict):
            if 'id' not in dataset:
                dataset = hdx.data.dataset.Dataset.read_from_hdx(dataset['name'])
            dataset = dataset['id']
        elif not isinstance(dataset, str):
            raise hdx.data.hdxobject.HDXError('Type %s cannot be added as a dataset!' % type(dataset).__name__)
        if is_valid_uuid(dataset) is False:
            raise hdx.data.hdxobject.HDXError('%s is not a valid dataset id!' % dataset)
        return {'showcase_id': self.data['id'], 'package_id': dataset}