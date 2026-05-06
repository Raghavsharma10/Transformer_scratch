def move(self, from_path, to_path, **kwargs):
        """移动单个文件或目录.

        :param from_path: 源文件/目录在网盘中的路径（包括文件名）。

                          .. warning::
                              * 路径长度限制为1000；
                              * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                              * 文件名或路径名开头结尾不能是 ``.``
                                或空白字符，空白字符包括：
                                ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param to_path: 目标文件/目录在网盘中的路径（包括文件名）。

                        .. warning::
                            * 路径长度限制为1000；
                            * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                            * 文件名或路径名开头结尾不能是 ``.``
                              或空白字符，空白字符包括：
                              ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :return: Response 对象
        """

        data = {
            'from': from_path,
            'to': to_path,
        }
        return self._request('file', 'move', data=data, **kwargs)