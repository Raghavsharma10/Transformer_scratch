def ext_publish(self, instance, loop, *args, **kwargs):
        """If 'external_signaller' is defined, calls it's publish method to
        notify external event systems.

        This is for internal usage only, but it's doumented because it's part
        of the interface with external notification systems.
        """
        if self.external_signaller is not None:
            # Assumes that the loop is managed by the external handler
            return self.external_signaller.publish_signal(self, instance, loop,
                                                          args, kwargs)