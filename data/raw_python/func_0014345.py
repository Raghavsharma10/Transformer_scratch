def base_url(klass, space_id, webhook_id, resource_id=None):
        """
        Returns the URI for the webhook call.
        """

        return "spaces/{0}/webhooks/{1}/calls/{2}".format(
            space_id,
            webhook_id,
            resource_id if resource_id is not None else ''
        )