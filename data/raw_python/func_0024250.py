def check_for_permission(self):
        # TODO: Works but not beautiful, needs review!
        """
        Checks if current user (or role) has the required permission
        for current workflow step.

        Raises:
            HTTPError: if user doesn't have required permissions.
        """
        if self.current.task:
            lane = self.current.lane_id
            permission = "%s.%s.%s" % (self.current.workflow_name, lane, self.current.task_name)
        else:
            permission = self.current.workflow_name
        log.debug("CHECK PERM: %s" % permission)

        if (self.current.task_type not in PERM_REQ_TASK_TYPES or
                permission.startswith(tuple(settings.ANONYMOUS_WORKFLOWS)) or
                (self.current.is_auth and permission.startswith(tuple(settings.COMMON_WORKFLOWS)))):
            return
        # FIXME:needs hardening

        log.debug("REQUIRE PERM: %s" % permission)
        if not self.current.has_permission(permission):
            raise HTTPError(403, "You don't have required permission: %s" % permission)