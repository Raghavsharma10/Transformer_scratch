def get_filtered_devices(
        self, model_name, device_types="upnp:rootdevice", timeout=2
    ):
        """
        returns a dict of devices that contain the given model name
        """

        # get list of all UPNP devices in the network
        upnp_devices = self.discover_upnp_devices(st=device_types)

        # go through all UPNP devices and filter wanted devices
        filtered_devices = collections.defaultdict(dict)
        for dev in upnp_devices.values():
            try:
                # download XML file with information about the device
                # from the device's location
                r = requests.get(dev.location, timeout=timeout)

                if r.status_code == requests.codes.ok:
                    # parse returned XML
                    root = ET.fromstring(r.text)

                    # add shortcut for XML namespace to access sub nodes
                    ns = {"upnp": "urn:schemas-upnp-org:device-1-0"}

                    # get device element
                    device = root.find("upnp:device", ns)

                    if model_name in device.find(
                        "upnp:modelName", ns
                    ).text:
                        # model name is wanted => add to list

                        # get unique UDN of the device that is used as key
                        udn = device.find("upnp:UDN", ns).text

                        # add url base
                        url_base = root.find("upnp:URLBase", ns)
                        if url_base is not None:
                            filtered_devices[udn][
                                "URLBase"
                            ] = url_base.text

                        # add interesting device attributes and
                        # use unique UDN as key
                        for attr in (
                            "deviceType", "friendlyName", "manufacturer",
                            "manufacturerURL", "modelDescription",
                            "modelName", "modelNumber"
                        ):
                            el = device.find("upnp:%s" % attr, ns)
                            if el is not None:
                                filtered_devices[udn][
                                    attr
                                ] = el.text.strip()

            except ET.ParseError:
                # just skip devices that are invalid xml
                pass
            except requests.exceptions.ConnectTimeout:
                # just skip devices that are not replying in time
                print("Timeout for '%s'. Skipping." % dev.location)

        return filtered_devices