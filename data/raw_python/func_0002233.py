def list_streams(self, file_type, start=0, limit=100,
                     filter_path=None, **kwargs):
        """以视频、音频、图片及文档四种类型的视图获取所创建应用程序下的
        文件列表.

        :param file_type: 类型分为video、audio、image及doc四种。
        :param start: 返回条目控制起始值，缺省值为0。
        :param limit: 返回条目控制长度，缺省为1000，可配置。
        :param filter_path: 需要过滤的前缀路径，如：/apps/album

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :return: Response 对象
        """

        params = {
            'type': file_type,
            'start': start,
            'limit': limit,
            'filter_path': filter_path,
        }
        return self._request('stream', 'list', extra_params=params,
                             **kwargs)