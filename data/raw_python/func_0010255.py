def send_command_block(self, target, command_block):
        """Send an arbitrary file system command block

        The primary use for this method is to send multiple file system commands with a single
        web service request.  This can help to avoid throttling.

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param command_block: The block of commands to execute on the target
        :type command_block: :class:`~FileSystemServiceCommandBlock`
        :return: The response will be a dictionary where the keys are device_ids and the values are
           the parsed responses of each command sent in the order listed in the command response for
           that device.  In practice it seems to be the same order as the commands were sent in, however,
           Device Cloud documentation does not explicitly state anywhere that is the case so I cannot
           guarantee it. This does mean that if you send different types of commands the response list
           will be different types.  Please see the commands parse_response functions for what those types
           will be. (:meth:`LsCommand.parse_response`, :class:`GetCommand.parse_response`,
           :class:`PutCommand.parse_response`, :class:`DeleteCommand.parse_response`)
        """
        root = _parse_command_response(
            self._sci_api.send_sci("file_system", target, command_block.get_command_string()))

        out_dict = {}
        for device in root.findall('./file_system/device'):
            device_id = device.get('id')
            results = []
            for command in device.find('./commands'):
                for command_class in FILE_SYSTEM_COMMANDS:
                    if command_class.command_name == command.tag.lower():
                        results.append(command_class.parse_response(command, fssapi=self, device_id=device_id))
            out_dict[device_id] = results
        return out_dict