def mkdir(self, remote_path, **kwargs):
        """为当前用户创建一个目录.

        :param remote_path: 网盘中目录的路径，必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :return: Response 对象
        """

        data = {
            'path': remote_path
        }
        return self._request('file', 'mkdir', data=data, **kwargs)