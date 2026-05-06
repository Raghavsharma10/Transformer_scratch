def rapid_upload(self, remote_path, content_length, content_md5,
                     content_crc32, slice_md5, ondup=None, **kwargs):
        """秒传一个文件.

        .. warning::
           * 被秒传文件必须大于256KB（即 256*1024 B）。
           * 校验段为文件的前256KB，秒传接口需要提供校验段的MD5。
             (非强一致接口，上传后请等待1秒后再读取)

        :param remote_path: 上传文件的全路径名。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param content_length: 待秒传文件的长度。
        :param content_md5: 待秒传文件的MD5。
        :param content_crc32: 待秒传文件的CRC32。
        :param slice_md5: 待秒传文件校验段的MD5。
        :param ondup: （可选）

                      * 'overwrite'：表示覆盖同名文件；
                      * 'newcopy'：表示生成文件副本并进行重命名，命名规则为“
                        文件名_日期.后缀”。
        :return: Response 对象
        """
        data = {
            'path': remote_path,
            'content-length': content_length,
            'content-md5': content_md5,
            'content-crc32': content_crc32,
            'slice-md5': slice_md5,
            'ondup': ondup,
        }
        return self._request('file', 'rapidupload', data=data, **kwargs)