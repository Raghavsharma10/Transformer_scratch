def delete(self, remote_path, **kwargs):
        """删除单个文件或目录.

        .. warning::
           * 文件/目录删除后默认临时存放在回收站内，删除文件或目录的临时存放
             不占用用户的空间配额；
           * 存放有效期为10天，10天内可还原回原路径下，10天后则永久删除。

        :param remote_path: 网盘中文件/目录的路径，路径必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :type remote_path: str
        :return: Response 对象
        """

        data = {
            'path': remote_path
        }
        return self._request('file', 'delete', data=data, **kwargs)