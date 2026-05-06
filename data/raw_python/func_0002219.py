def upload_tmpfile(self, file_content, **kwargs):
        """分片上传—文件分片及上传.

        百度 PCS 服务支持每次直接上传最大2G的单个文件。

        如需支持上传超大文件（>2G），则可以通过组合调用分片文件上传的
        ``upload_tmpfile`` 方法和 ``upload_superfile`` 方法实现：

        1. 首先，将超大文件分割为2G以内的单文件，并调用 ``upload_tmpfile``
           将分片文件依次上传；
        2. 其次，调用 ``upload_superfile`` ，完成分片文件的重组。

        除此之外，如果应用中需要支持断点续传的功能，
        也可以通过分片上传文件并调用 ``upload_superfile`` 接口的方式实现。

        :param file_content: 上传文件的内容/文件对象
                             (e.g. ``open('foobar', 'rb')`` )
        :return: Response 对象
        """

        params = {
            'type': 'tmpfile'
        }
        files = {'file': ('file', file_content, '')}
        url = 'https://c.pcs.baidu.com/rest/2.0/pcs/file'
        return self._request('file', 'upload', url=url, extra_params=params,
                             files=files, **kwargs)