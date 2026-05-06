def get_container_image_ids(self):
        """
        Find the image IDs the containers use.

        :return: dict, image tag to docker ID
        """

        statuses = graceful_chain_get(self.json, "status", "containerStatuses")
        if statuses is None:
            return {}

        def remove_prefix(image_id, prefix):
            if image_id.startswith(prefix):
                return image_id[len(prefix):]

            return image_id

        return {status['image']: remove_prefix(status['imageID'], 'docker://')
                for status in statuses}