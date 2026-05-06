def create_folder(self, dir_name):
        """创建目录(https://www.qcloud.com/document/product/436/6061)

        :param dir_name:要创建的目录的目录的名称
        :return 返回True创建成功，返回False创建失败
        """
        if dir_name[0] == '/':
            dir_name = dir_name[1:len(dir_name)]
        self.url = "http://<Region>.file.myqcloud.com" + "/files/v2/<appid>/<bucket_name>/<dir_name>/"
        self.url = self.url.replace("<Region>", self.config.region).replace("<appid>", str(self.config.app_id))
        self.url = str(self.url).replace("<bucket_name>", self.config.bucket).replace("<dir_name>", dir_name)
        self.headers['Authorization'] = CosAuth(self.config).sign_more(self.config.bucket, '', 30 )
        response, content = self.http.request(uri=self.url, method='POST', body='{"op": "create", "biz_attr": ""}', headers=self.headers)
        if eval(content.decode('utf8')).get("code") == 0:
            return True
        else:
            return False