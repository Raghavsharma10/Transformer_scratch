def get_file(self, target, path, offset=None, length=None):
        """Get the contents of a file on the device

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to the file to retrieve
        :param offset: Start retrieving data from this byte position in the file, if None start from the beginning
        :param length: How many bytes to retrieve, if None retrieve until the end of the file
        :return: A dictionary with keys of device ids and values of the bytes of the file (or partial file if offset
            and/or length are specified) or an :class:`~.ErrorInfo` object if there was an error response
        :raises: :class:`~.ResponseParseError` If the SCI response has unrecognized formatting
        """
        command_block = FileSystemServiceCommandBlock()
        command_block.add_command(GetCommand(path, offset, length))
        root = _parse_command_response(
            self._sci_api.send_sci("file_system", target, command_block.get_command_string()))
        out_dict = {}
        for device in root.findall('./file_system/device'):
            device_id = device.get('id')
            error = device.find('./error')
            if error is not None:
                out_dict[device_id] = _parse_error_tree(error)
            else:
                data = GetCommand.parse_response(device.find('./commands/get_file'))
                out_dict[device_id] = data
        return out_dict