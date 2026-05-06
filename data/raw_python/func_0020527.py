def _get_image_stream_info_for_build_request(self, build_request):
        """Return ImageStream, and ImageStreamTag name for base_image of build_request

        If build_request is not auto instantiated, objects are not fetched
        and None, None is returned.
        """
        image_stream = None
        image_stream_tag_name = None

        if build_request.has_ist_trigger():
            image_stream_tag_id = build_request.trigger_imagestreamtag
            image_stream_id, image_stream_tag_name = image_stream_tag_id.split(':')

            try:
                image_stream = self.get_image_stream(image_stream_id).json()
            except OsbsResponseException as x:
                if x.status_code != 404:
                    raise

            if image_stream:
                try:
                    self.get_image_stream_tag(image_stream_tag_id).json()
                except OsbsResponseException as x:
                    if x.status_code != 404:
                        raise

        return image_stream, image_stream_tag_name