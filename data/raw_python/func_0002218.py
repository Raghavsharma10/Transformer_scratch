def upload(self, remote_path, file_content, ondup=None, **kwargs):
        """上传单个文件（<2G）.

        | 百度PCS服务目前支持最大2G的单个文件上传。
        | 如需支持超大文件（>2G）的断点续传，请参考下面的“分片文件上传”方法。

        :param remote_path: 网盘中文件的保存路径（包含文件名）。
                            必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param file_content: 上传文件的内容/文件对象 。
                             (e.g. ``open('foobar', 'rb')`` )
        :param ondup: （可选）

                      * 'overwrite'：表示覆盖同名文件；
                      * 'newcopy'：表示生成文件副本并进行重命名，命名规则为“
                        文件名_日期.后缀”。
        :return: Response 对象
        """

        params = {
            'path': remote_path,
            'ondup': ondup
        }
        files = {'file': ('file', file_content, '')}
        url = 'https://c.pcs.baidu.com/rest/2.0/pcs/file'
        return self._request('file', 'upload', url=url, extra_params=params,
                             files=files, **kwargs)