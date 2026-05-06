def check_for_authentication(self):
        """
        Checks current workflow against :py:data:`~zengine.settings.ANONYMOUS_WORKFLOWS` list.

        Raises:
            HTTPUnauthorized: if WF needs an authenticated user and current user isn't.
        """
        auth_required = self.current.workflow_name not in settings.ANONYMOUS_WORKFLOWS
        if auth_required and not self.current.is_auth:
            self.current.log.debug("LOGIN REQUIRED:::: %s" % self.current.workflow_name)
            raise HTTPError(401, "Login required for %s" % self.current.workflow_name)