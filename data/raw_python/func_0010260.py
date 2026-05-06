def get_modified_items(self, target, path, last_modified_cutoff):
        """Get all files and directories from a path on the device modified since a given time

        :param target: The device(s) to be targeted with this request
        :type target: :class:`devicecloud.sci.TargetABC` or list of :class:`devicecloud.sci.TargetABC` instances
        :param path: The path on the target to the directory to check for modified files.
        :param last_modified_cutoff: The time (as Unix epoch time) to get files modified since
        :type last_modified_cutoff: int
        :return: A dictionary where the key is a device id and the value is either an :class:`~.ErrorInfo` if there
            was a problem with the operation or a :class:`~.LsInfo` with the items modified since the
            specified date
        """
        file_list = self.list_files(target, path)
        out_dict = {}
        for device_id, device_data in six.iteritems(file_list):
            if isinstance(device_data, ErrorInfo):
                out_dict[device_id] = device_data
            else:
                files = []
                dirs = []
                for cur_file in device_data.files:
                    if cur_file.last_modified > last_modified_cutoff:
                        files.append(cur_file)

                for cur_dir in device_data.directories:
                    if cur_dir.last_modified > last_modified_cutoff:
                        dirs.append(cur_dir)
                out_dict[device_id] = LsInfo(directories=dirs, files=files)

        return out_dict