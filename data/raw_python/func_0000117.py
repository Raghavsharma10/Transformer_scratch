def get_activitylog(self, after=None, severity=None, start=None, end=None):
        """
        Returns activitylog object
        severity - filter severity ('INFO', DEBUG')
        start/end - time or log text

        """
        if after:
            log_raw = self._router.get_instance_activitylog(org_id=self.organizationId,
                                                            instance_id=self.instanceId,
                                                            params={"after": after}).json()
        else:
            log_raw = self._router.get_instance_activitylog(org_id=self.organizationId,
                                                            instance_id=self.instanceId).json()

        return ActivityLog(log_raw, severity=severity, start=start, end=end)