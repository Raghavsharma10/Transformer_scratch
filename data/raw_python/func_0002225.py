def multi_move(self, path_list, **kwargs):
        """批量移动文件或目录.

        :param path_list: 源文件地址和目标文件地址对列表:

                          >>> path_list = [
                          ...   ('/apps/test_sdk/test.txt',  # 源文件
                          ...    '/apps/test_sdk/testmkdir/b.txt'  # 目标文件
                          ...   ),
                          ...   ('/apps/test_sdk/test.txt',  # 源文件
                          ...    '/apps/test_sdk/testmkdir/b.txt'  # 目标文件
                          ...   ),
                          ... ]

                          .. warning::
                              * 路径长度限制为1000；
                              * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                              * 文件名或路径名开头结尾不能是 ``.``
                                或空白字符，空白字符包括：
                                ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :type path_list: list
        :return: Response 对象
        """

        data = {
            'param': json.dumps({
                'list': [{'from': x[0], 'to': x[1]} for x in path_list]
            }),
        }
        return self._request('file', 'move', data=data, **kwargs)