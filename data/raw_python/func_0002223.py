def list_files(self, remote_path, by=None, order=None,
                   limit=None, **kwargs):
        """获取目录下的文件列表.

        :param remote_path: 网盘中目录的路径，必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param by: 排序字段，缺省根据文件类型排序：

                   * time（修改时间）
                   * name（文件名）
                   * size（大小，注意目录无大小）
        :param order: “asc”或“desc”，缺省采用降序排序。

                      * asc（升序）
                      * desc（降序）
        :param limit: 返回条目控制，参数格式为：n1-n2。

                      返回结果集的[n1, n2)之间的条目，缺省返回所有条目；
                      n1从0开始。
        :return: Response 对象
        """

        params = {
            'path': remote_path,
            'by': by,
            'order': order,
            'limit': limit
        }
        return self._request('file', 'list', extra_params=params, **kwargs)