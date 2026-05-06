def alter(self, interfaces):
        """
        Used to provide the ability to alter the interfaces dictionary before
        it is returned from self.parse().

        Required Arguments:

            interfaces
                The interfaces dictionary.

        Returns: interfaces dict

        """
        # fixup some things
        for device, device_dict in interfaces.items():
            if len(device_dict['inet4']) > 0:
                device_dict['inet'] = device_dict['inet4'][0]
            if 'inet' in device_dict and not device_dict['inet'] is None:
                try:
                    host = socket.gethostbyaddr(device_dict['inet'])[0]
                    interfaces[device]['hostname'] = host
                except (socket.herror, socket.gaierror):
                    interfaces[device]['hostname'] = None

            # To be sure that hex values and similar are always consistent, we
            # return everything in lowercase. For instance, Windows writes
            # MACs in upper-case.
            for key, device_item in device_dict.items():
                if hasattr(device_item, 'lower'):
                    interfaces[device][key] = device_dict[key].lower()

        return interfaces