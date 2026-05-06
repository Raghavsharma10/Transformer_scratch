def put_file(self, target, path, file_data=None, server_file=None, offset=None, truncate=False):
        """Put data into a file on the device

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to the file to write to.  If the file already exists it will be overwritten.
        :param file_data: A `six.binary_type` containing the data to put into the file
        :param server_file: The path to a file on the devicecloud server containing the data to put into the file on the
            device
        :param offset: Start writing bytes to the file at this position, if None start at the beginning
        :param truncate: Boolean, if True after bytes are done being written end the file their even if previous data
            exists beyond it.  If False, leave any existing data in place.
        :return: A dictionary with keys being device ids and value being None if successful or an :class:`~.ErrorInfo`
            if the operation failed on that device
        :raises: :class:`~.FileSystemServiceException` if either both file_data and server_file are specified or
            neither are specified
        :raises: :class:`~.ResponseParseError` If the SCI response has unrecognized formatting
        """

        command_block = FileSystemServiceCommandBlock()
        command_block.add_command(PutCommand(path, file_data, server_file, offset, truncate))

        root = _parse_command_response(self._sci_api.send_sci("file_system", target, command_block.get_command_string()))
        out_dict = {}
        for device in root.findall('./file_system/device'):
            device_id = device.get('id')
            error = device.find('./error')
            if error is not None:
                out_dict[device_id] = _parse_error_tree(error)
            else:
                out_dict[device_id] = PutCommand.parse_response(device.find('./commands/put_file'))

        return out_dict