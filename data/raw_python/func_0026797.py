def add_device(self, name, protocol, model=None, **parameters):
        """Add a new device.

        :return: a :class:`Device` or :class:`DeviceGroup` instance.
        """
        device = Device(self.lib.tdAddDevice(), lib=self.lib)
        try:
            device.name = name
            device.protocol = protocol
            if model:
                device.model = model
            for key, value in parameters.items():
                device.set_parameter(key, value)

            # Return correct type
            return DeviceFactory(device.id, lib=self.lib)
        except Exception:
            import sys
            exc_info = sys.exc_info()
            try:
                device.remove()
            except:
                pass

            if "with_traceback" in dir(Exception):
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
            else:
                exec("raise exc_info[0], exc_info[1], exc_info[2]")