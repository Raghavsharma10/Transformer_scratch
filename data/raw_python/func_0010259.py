def delete_file(self, target, path):
        """Delete a file from a device

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to the file to delete.
        :return: A dictionary with keys being device ids and value being None if successful or an :class:`~.ErrorInfo`
            if the operation failed on that device
        :raises: :class:`~.ResponseParseError` If the SCI response has unrecognized formatting
        """
        command_block = FileSystemServiceCommandBlock()
        command_block.add_command(DeleteCommand(path))
        root = _parse_command_response(self._sci_api.send_sci("file_system", target, command_block.get_command_string()))

        out_dict = {}
        for device in root.findall('./file_system/device'):
            device_id = device.get('id')
            error = device.find('./error')
            if error is not None:
                out_dict[device_id] = _parse_error_tree(error)
            else:
                out_dict[device_id] = DeleteCommand.parse_response(device.find('./commands/rm'))
        return out_dict