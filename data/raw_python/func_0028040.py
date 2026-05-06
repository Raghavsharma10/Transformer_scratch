def _write_to_hdx(self, action, data, id_field_name, file_to_upload=None):
        # type: (str, Dict, str, Optional[str]) -> Dict
        """Creates or updates an HDX object in HDX and return HDX object metadata dict

        Args:
            action (str): Action to perform eg. 'create', 'update'
            data (Dict): Data to write to HDX
            id_field_name (str): Name of field containing HDX object identifier or None
            file_to_upload (Optional[str]): File to upload to HDX

        Returns:
            Dict: HDX object metadata
        """
        file = None
        try:
            if file_to_upload:
                file = open(file_to_upload, 'rb')
                files = [('upload', file)]
            else:
                files = None
            return self.configuration.call_remoteckan(self.actions()[action], data, files=files)
        except Exception as e:
            raisefrom(HDXError, 'Failed when trying to %s %s! (POST)' % (action, data[id_field_name]), e)
        finally:
            if file_to_upload and file:
                file.close()