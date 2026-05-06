def get_feature(self, croplayer_id, cropfeature_id):
        """
        Gets a crop feature

        :param int croplayer_id: ID of a cropping layer
        :param int cropfeature_id: ID of a cropping feature
        :rtype: CropFeature
        """
        target_url = self.client.get_url('CROPFEATURE', 'GET', 'single', {'croplayer_id': croplayer_id, 'cropfeature_id': cropfeature_id})
        return self.client.get_manager(CropFeature)._get(target_url)