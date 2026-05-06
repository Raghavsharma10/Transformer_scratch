def download_stream(self, remote_path, **kwargs):
        """为当前用户下载一个流式文件.其参数和返回结果与下载单个文件的相同.

        :param remote_path: 需要下载的文件路径，以/开头的绝对路径，含文件名。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :return: Response 对象
        """

        params = {
            'path': remote_path,
        }
        url = 'https://d.pcs.baidu.com/rest/2.0/pcs/file'
        return self._request('stream', 'download', url=url,
                             extra_params=params, **kwargs)