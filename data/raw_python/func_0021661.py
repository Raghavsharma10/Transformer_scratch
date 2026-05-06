def commit_config(self):
        """
        Netmiko is being used to commit the configuration because it takes
        a better care of results compared to pan-python.
        """
        if self.loaded:
            if self.ssh_connection is False:
                self._open_ssh()
            try:
                self.ssh_device.commit()
                time.sleep(3)
                self.loaded = False
                self.changed = True
            except:   # noqa
                if self.merge_config:
                    raise MergeConfigException('Error while commiting config')
                else:
                    raise ReplaceConfigException('Error while commiting config')
        else:
            raise ReplaceConfigException('No config loaded.')