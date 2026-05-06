def is_stopped(self, *args, **kwargs):
        """Return whether this container is stopped"""
        kwargs["waiting"] = False
        return self.wait_till_stopped(*args, **kwargs)