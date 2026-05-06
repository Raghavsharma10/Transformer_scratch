def exists(self, target, path, path_sep="/"):
        """Check if path refers to an existing path on the device

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to check for existence.
        :param path_sep: The path separator of the device
        :return: A dictionary where the key is a device id and the value is either an :class:`~.ErrorInfo` if there
            was a problem with the operation or a boolean with the existence status of the path on that device
        """
        if path.endswith(path_sep):
            path = path[:-len(path_sep)]
        par_dir, filename = path.rsplit(path_sep, 1)
        file_list = self.list_files(target, par_dir)
        out_dict = {}
        for device_id, device_data in six.iteritems(file_list):
            if isinstance(device_data, ErrorInfo):
                out_dict[device_id] = device_data
            else:
                out_dict[device_id] = False
                for cur_file in device_data.files:
                    if cur_file.path == path:
                        out_dict[device_id] = True
                for cur_dir in device_data.directories:
                    if cur_dir.path == path:
                        out_dict[device_id] = True

        return out_dict