def _get_assets(self, bbox, size=None, page=None, asset_type=None,
            device_type=None, event_type=None, media_type=None):
        """
        Returns the raw results of an asset search for a given bounding
        box.
        """
        uri = self.uri + '/v1/assets/search'
        headers = self._get_headers()

        params = {
                'bbox': bbox,
                }

        # Query parameters

        params['q'] = []
        if device_type:
            if isinstance(device_type, str):
                device_type = [device_type]

            for device in device_type:
                if device not in self.DEVICE_TYPES:
                    logging.warning("Invalid device type: %s" % device)

                params['q'].append("device-type:%s" % device)

        if asset_type:
            if isinstance(asset_type, str):
                asset_type = [asset_type]

            for asset in asset_type:
                if asset not in self.ASSET_TYPES:
                    logging.warning("Invalid asset type: %s" % asset)
                params['q'].append("assetType:%s" % asset)

        if media_type:
            if isinstance(media_type, str):
                media_type = [media_type]

            for media in media_type:
                if media not in self.MEDIA_TYPES:
                    logging.warning("Invalid media type: %s" % media)
                params['q'].append("mediaType:%s" % media)

        if event_type:
            if isinstance(event_type, str):
                event_type = [event_type]

            for event in event_type:
                if event not in self.EVENT_TYPES:
                    logging.warning("Invalid event type: %s" % event)
                params['q'].append("eventTypes:%s" % event)

        # Pagination parameters

        if size:
            params['size'] = size

        if page:
            params['page'] = page

        return self.service._get(uri, params=params, headers=headers)