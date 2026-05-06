def upload(self, file_path, dataset=None, public=False):
        """Use this function to upload data to Knoema dataset."""

        upload_status = self.upload_file(file_path)
        err_msg = 'Dataset has not been uploaded to the remote host'
        if not upload_status.successful:
            msg = '{}, because of the following error: {}'.format(err_msg, upload_status.error)
            raise ValueError(msg)

        err_msg = 'File has not been verified'
        upload_ver_status = self.upload_verify(upload_status.properties.location, dataset)
        if not upload_ver_status.successful:
            ver_err = '\r\n'.join(upload_ver_status.errors)
            msg = '{}, because of the following error(s): {}'.format(err_msg, ver_err)
            raise ValueError(msg)

        ds_upload = definition.DatasetUpload(upload_ver_status, upload_status, dataset, public)
        ds_upload_submit_result = self.upload_submit(ds_upload)
        err_msg = 'Dataset has not been saved to the database'
        if ds_upload_submit_result.status == 'failed':
            ver_err = '\r\n'.join(ds_upload_submit_result.errors)
            msg = '{}, because of the following error(s): {}'.format(err_msg, ver_err)
            raise ValueError(msg)

        ds_upload_result = None
        while True:
            ds_upload_result = self.upload_status(ds_upload_submit_result.submit_id)
            if ds_upload_result.status == 'pending' or ds_upload_result.status == 'processing':
                time.sleep(5)
            else:
                break

        if ds_upload_result.status != 'successful':
            ver_err = '\r\n'.join(ds_upload_result.errors)
            msg = '{}, because of the following error(s): {}'.format(err_msg, ver_err)
            raise ValueError(msg)

        return ds_upload_result.dataset