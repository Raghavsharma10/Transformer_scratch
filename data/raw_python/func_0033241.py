def upload_slice_file(self, real_file_path, slice_size, file_name, offset=0, dir_name=None):
        """
        此分片上传代码由GitHub用户a270443177(https://github.com/a270443177)友情提供

        :param real_file_path:
        :param slice_size:
        :param file_name:
        :param offset:
        :param dir_name:
        :return:
        """
        if dir_name is not None and dir_name[0] == '/':
            dir_name = dir_name[1:len(dir_name)]
        if dir_name is None:
            dir_name = ""
        self.url = 'http://' + self.config.region + '.file.myqcloud.com/files/v2/' + str(
            self.config.app_id) + '/' + self.config.bucket
        if dir_name is not None:
            self.url = self.url + '/' + dir_name
        self.url = self.url + '/' + file_name
        file_size = os.path.getsize(real_file_path)
        session = self._upload_slice_control(file_size=file_size, slice_size=slice_size)
        with open(real_file_path, 'rb') as local_file:
            while offset < file_size:
                file_content = local_file.read(slice_size)
                self._upload_slice_data(filecontent=file_content, session=session, offset=offset)
                offset += slice_size
            r = self._upload_slice_finish(session=session, file_size=file_size)
        return r