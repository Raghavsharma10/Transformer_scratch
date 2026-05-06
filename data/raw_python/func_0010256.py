def list_files(self, target, path, hash='any'):
        """List all files and directories in the path on the target

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to list files and directories from
        :param hash: an optional attribute which indicates a hash over the file contents should be retrieved. Values
            include none, any, md5, and crc32. any is used to indicate the device should choose its best available hash.
        :return: A dictionary with keys of device ids and values of :class:`~.LsInfo` objects containing the files and
            directories or an :class:`~.ErrorInfo` object if there was an error response
        :raises: :class:`~.ResponseParseError` If the SCI response has unrecognized formatting

        Here is an example usage::

            # dc is a DeviceCloud instance
            fssapi = dc.get_fss_api()

            target = AllTarget()
            ls_dir = '/root/home/user/important_files/'

            ls_data = fssapi.list_files(target, ls_dir)

            # Loop over all device results
            for device_id, device_data in ls_data.iteritems():
                # Check if it succeeded or was an error
                if isinstance(device_data, ErrorInfo):
                    # Do some error handling
                    logger.warn("Error listing file info on device %s. errno: %s message:%s",
                                device_id, device_data.errno, device_data.message)

                # It's of type LsInfo
                else:
                    # Look at all the files
                    for finfo in device_data.files:
                        logger.info("Found file %s of size %s on device %s",
                                    finfo.path, finfo.size, device_id)
                    # Look at all the directories
                    for dinfo in device_data.directories:
                        logger.info("Found directory %s of last modified %s on device %s",
                                    dinfo.path, dinfo.last_modified, device_id)

        """
        command_block = FileSystemServiceCommandBlock()
        command_block.add_command(LsCommand(path, hash=hash))
        root = _parse_command_response(
            self._sci_api.send_sci("file_system", target, command_block.get_command_string()))

        out_dict = {}

        #  At this point the XML we have is of the form
        # <sci_reply>
        #   <file_system>
        #     <device id="device_id">
        #       <commands>
        #         <ls hash="hash_type">
        #           <file path="file_path" last_modified=last_modified_time ... />
        #           ...
        #           <dir path="dir_path" last_modified=last_modified_time />
        #           ...
        #         </ls>
        #       </commands>
        #     </device>
        #     <device id="device_id">
        #       <commands>
        #         <ls hash="hash_type">
        #           <file path="file_path" last_modified=last_modified_time ... />
        #           ...
        #           <dir path="dir_path" last_modified=last_modified_time />
        #           ...
        #         </ls>
        #       </commands>
        #     </device>
        #     ...
        #   </file_system>
        # </sci_reply>

        # Here we will get each of the XML trees rooted at the device nodes
        for device in root.findall('./file_system/device'):
            device_id = device.get('id')
            error = device.find('./error')
            if error is not None:
                out_dict[device_id] = _parse_error_tree(error)
            else:
                linfo = LsCommand.parse_response(device.find('./commands/ls'), device_id=device_id, fssapi=self)
                out_dict[device_id] = linfo
        return out_dict