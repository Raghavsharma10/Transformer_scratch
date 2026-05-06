def search(self, remote_path, keyword, recurrent='0', **kwargs):
        """按文件名搜索文件（不支持查找目录）.

        :param remote_path: 需要检索的目录路径，路径必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :type remote_path: str
        :param keyword: 关键词
        :type keyword: str
        :param recurrent: 是否递归。

                          * "0"表示不递归
                          * "1"表示递归
        :type recurrent: str
        :return: Response 对象
        """

        params = {
            'path': remote_path,
            'wd': keyword,
            're': recurrent,
        }
        return self._request('file', 'search', extra_params=params, **kwargs)