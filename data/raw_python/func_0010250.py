def delete(self):
        """Delete this file from the device

        .. note::
           After deleting the file, this object will no longer contain valid information
           and further calls to delete or get_data will return :class:`~.ErrorInfo` objects
        """
        target = DeviceTarget(self.device_id)
        return self._fssapi.delete_file(target, self.path)[self.device_id]