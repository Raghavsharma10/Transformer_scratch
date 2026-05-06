def scale_app(self, app_id, instances=None, delta=None, force=False):
        """Scale an app.

        Scale an app to a target number of instances (with `instances`), or scale the number of
        instances up or down by some delta (`delta`). If the resulting number of instances would be negative,
        desired instances will be set to zero.

        If both `instances` and `delta` are passed, use `instances`.

        :param str app_id: application ID
        :param int instances: [optional] the number of instances to scale to
        :param int delta: [optional] the number of instances to scale up or down by
        :param bool force: apply even if a deployment is in progress

        :returns: a dict containing the deployment id and version
        :rtype: dict
        """
        if instances is None and delta is None:
            marathon.log.error('instances or delta must be passed')
            return

        try:
            app = self.get_app(app_id)
        except NotFoundError:
            marathon.log.error('App "{app}" not found'.format(app=app_id))
            return

        desired = instances if instances is not None else (
            app.instances + delta)
        return self.update_app(app.id, MarathonApp(instances=desired), force=force)