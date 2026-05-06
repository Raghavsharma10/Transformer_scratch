def upload_file(self, real_file_path, file_name, dir_name=None):
        """简单上传文件(https://www.qcloud.com/document/product/436/6066)

        :param real_file_path: 文件的物理地址
        :param file_name: 文件名称
        :param dir_name: 文件夹名称（可选）
        :return:json数据串
        """

        if dir_name is not None and dir_name[0] == '/':
            dir_name = dir_name[1:len(dir_name)]
        if dir_name is None:
            dir_name = ""
        self.url = 'http://' + self.config.region + '.file.myqcloud.com/files/v2/' + str(self.config.app_id) + '/' + self.config.bucket
        if dir_name is not None:
            self.url = self.url + '/' + dir_name
        self.url = self.url + '/' + file_name
        headers = {}
        headers['Authorization'] = CosAuth(self.config).sign_more(self.config.bucket, '', 30)
        files = {'file': ('', open(real_file_path, 'rb'))}
        r = requests.post(url=self.url, data={'op': 'upload', 'biz_attr': '', 'insertOnly': '0'}, files={
            'filecontent': (real_file_path, open(real_file_path, 'rb'), 'application/octet-stream')}, headers=headers)
        return str(eval(r.content.decode('utf8')).get('data'))