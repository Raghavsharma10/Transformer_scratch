def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests.
        Expects the 'vid' of the version to act on
        to be passed as in the POST variable 'version'.

        If a POST variable 'revert' is present this will
        call the revert method and then return a 'render
        redirect' to the result of the `get_done_url` method.

        If a POST variable 'delete' is present this will
        call the delete method and return a 'render redirect'
        to the result of the `get_done_url` method.

        If this method receives unexpected input, it will
        silently redirect to the result of the `get_done_url`
        method.
        """

        versions = self._get_versions()
        url = self.get_done_url()
        msg = None

        try:
            vid = int(request.POST.get('version', ''))
            version = versions.get(vid=vid)
            if request.POST.get('revert'):
                object_url = self.get_object_url()
                msg = self.revert(version, object_url)
            elif request.POST.get('delete'):
                msg = self.delete(version)
                # Delete should redirect back to itself
                url = self.request.build_absolute_uri()

        # If the give version isn't valid we'll just silently redirect
        except (ValueError, versions.model.DoesNotExist):
            pass

        return self.render(request, redirect_url=url,
                   message=msg,
                   obj=self.object,
                   collect_render_data=False)