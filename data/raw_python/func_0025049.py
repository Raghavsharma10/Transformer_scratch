def get_assets(self, bbox, **kwargs):
        """
        Query the assets stored in the intelligent environment for a given
        bounding box and query.

        Assets can be filtered by type of asset, event, or media available.

            - device_type=['DATASIM']
            - asset_type=['CAMERA']
            - event_type=['PKIN']
            - media_type=['IMAGE']

        Pagination can be controlled with keyword parameters

            - page=2
            - size=100

        Returns a list of assets stored in a dictionary that describe their:

            - asset-id
            - device-type
            - device-id
            - media-type
            - coordinates
            - event-type

        Additionally there are some _links for additional information.
        """
        response = self._get_assets(bbox, **kwargs)

        # Remove broken HATEOAS _links but identify asset uid first
        assets = []
        for asset in response['_embedded']['assets']:
            asset_url = asset['_links']['self']
            uid = asset_url['href'].split('/')[-1]
            asset['uid'] = uid

            del(asset['_links'])
            assets.append(asset)

        return assets