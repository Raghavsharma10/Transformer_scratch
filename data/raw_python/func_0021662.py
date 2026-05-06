def rollback(self):
        """
        Netmiko is being used to commit the rollback configuration because
        it takes a better care of results compared to pan-python.
        """
        if self.changed:
            rollback_cmd = '<load><config><from>{0}</from></config></load>'.format(self.backup_file)
            self.device.op(cmd=rollback_cmd)
            time.sleep(5)

            if self.ssh_connection is False:
                self._open_ssh()
            try:
                self.ssh_device.commit()
                self.loaded = False
                self.changed = False
                self.merge_config = False
            except:    # noqa
                ReplaceConfigException("Error while loading backup config")