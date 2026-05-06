def list_folder(self, dir_name=None, prefix=None, num=1000, context=None):
        """列目录(https://www.qcloud.com/document/product/436/6062)

        :param dir_name:文件夹名称
        :param prefix:前缀
        :param num:查询的文件的数量，最大支持1000，默认查询数量为1000
        :param context:翻页标志，将上次查询结果的context的字段传入，即可实现翻页的功能
        :return 查询结果，为json格式
        """
        if dir_name[0] == '/':
            dir_name = dir_name[1:len(dir_name)]
        self.url = 'http://<Region>.file.myqcloud.com/files/v2/<appid>/<bucket_name>/'
        self.url = self.url.replace("<Region>", self.config.region).replace("<appid>", str(self.config.app_id)).replace("<bucket_name>", self.config.bucket)
        if dir_name is not None:
            self.url = self.url + str(dir_name) + "/"
        if prefix is not None:
            self.url = self.url + str(prefix)
        self.url = self.url + "?op=list&num=" + str(num)
        if context is not None:
            self.url = self.url + '&context=' + str(context)
        self.headers['Authorization'] = CosAuth(self.config).sign_more(self.config.bucket, '', 30)
        response, content = self.http.request(uri=self.url, method='GET', headers=self.headers)
        return content.decode("utf8")