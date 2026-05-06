def query_folder(self, dir_name):
        """查询目录属性(https://www.qcloud.com/document/product/436/6063)

        :param dir_name:查询的目录的名称
        :return:查询出来的结果，为json格式
        """
        if dir_name[0] == '/':
            dir_name = dir_name[1:len(dir_name)]
        self.url = 'http://' + self.config.region + '.file.myqcloud.com' + '/files/v2/' + str(self.config.app_id) + '/' + self.config.bucket + '/' + dir_name + '/?op=stat'
        self.headers['Authorization'] = CosAuth(self.config).sign_more(self.config.bucket, '', 30)
        reponse, content = self.http.request(uri=self.url, method='GET',headers=self.headers)
        return content.decode("utf8")