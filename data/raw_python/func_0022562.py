def _cpu(self):
        """Record CPU usage."""
        value = int(psutil.cpu_percent())
        set_metric("cpu", value, category=self.category)
        gauge("cpu", value)