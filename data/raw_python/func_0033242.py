def upload_file_from_url(self, url, file_name, dir_name=None):
        """简单上传文件(https://www.qcloud.com/document/product/436/6066)

        :param url: 文件url地址
        :param file_name: 文件名称
        :param dir_name: 文件夹名称（可选）
        :return:json数据串
        """
        real_file_name = str(int(time.time()*1000))
        urllib.request.urlretrieve(url, real_file_name)
        data = self.upload_file(real_file_name, file_name, dir_name)
        os.remove(real_file_name)
        return data