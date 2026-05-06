def provision_devices(self, devices):
        """Provision multiple devices with a single API call

        This method takes an iterable of dictionaries where the values in the dictionary are
        expected to match the arguments of a call to :meth:`provision_device`.  The
        contents of each dictionary will be validated.

        :param list devices: An iterable of dictionaries each containing information about
            a device to be provision.  The form of the dictionary should match the keyword
            arguments taken by :meth:`provision_device`.
        :raises DeviceCloudHttpException: If there is an unexpected error reported by Device Cloud.
        :raises ValueError: If any input fields are known to have a bad form.
        :return: A list of dictionaries in the form described for :meth:`provision_device` in the
            order matching the requested device list.  Note that it is possible for there to
            be mixed success and error when provisioning multiple devices.

        """
        # Validate all the input for each device provided
        sio = six.StringIO()

        def write_tag(tag, val):
            sio.write("<{tag}>{val}</{tag}>".format(tag=tag, val=val))

        def maybe_write_element(tag, val):
            if val is not None:
                write_tag(tag, val)
                return True
            return False

        sio.write("<list>")
        for d in devices:
            sio.write("<DeviceCore>")

            mac_address = d.get("mac_address")
            device_id = d.get("device_id")
            imei = d.get("imei")
            if mac_address is not None:
                write_tag("devMac", mac_address)
            elif device_id is not None:
                write_tag("devConnectwareId", device_id)
            elif imei is not None:
                write_tag("devCellularModemId", imei)
            else:
                raise ValueError("mac_address, device_id, or imei must be provided for device %r" % d)

            # Write optional elements if present.
            maybe_write_element("grpPath", d.get("group_path"))
            maybe_write_element("dpUserMetaData", d.get("metadata"))
            maybe_write_element("dpTags", d.get("tags"))
            maybe_write_element("dpMapLong", d.get("map_long"))
            maybe_write_element("dpMapLat", d.get("map_lat"))
            maybe_write_element("dpContact", d.get("contact"))
            maybe_write_element("dpDescription", d.get("description"))

            sio.write("</DeviceCore>")
        sio.write("</list>")

        # Send the request, set the Accept XML as a nicety
        results = []
        response = self._conn.post("/ws/DeviceCore", sio.getvalue(), headers={'Accept': 'application/xml'})
        root = ET.fromstring(response.content)  # <result> tag is root of <list> response
        for child in root:
            if child.tag.lower() == "location":
                results.append({
                    "error": False,
                    "error_msg": None,
                    "location": child.text
                })
            else:  # we expect "error" but handle generically
                results.append({
                    "error": True,
                    "location": None,
                    "error_msg": child.text
                })

        return results