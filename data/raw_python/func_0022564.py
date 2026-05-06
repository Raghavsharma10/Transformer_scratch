def _disk(self):
        """Record Disk usage."""
        mountpoints = [
            p.mountpoint for p in psutil.disk_partitions()
            if p.device.endswith(self.device)
        ]
        if len(mountpoints) != 1:
            raise CommandError("Unknown device: {0}".format(self.device))

        value = int(psutil.disk_usage(mountpoints[0]).percent)
        set_metric("disk-{0}".format(self.device), value, category=self.category)
        gauge("disk-{0}".format(self.device), value)