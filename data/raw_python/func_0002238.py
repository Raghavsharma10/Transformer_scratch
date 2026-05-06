def list_download_tasks(self, need_task_info=1, start=0, limit=10, asc=0,
                            create_time=None, status=None, source_url=None,
                            remote_path=None, expires=None, **kwargs):
        """查询离线下载任务ID列表及任务信息.

        :param need_task_info: 是否需要返回任务信息:

                               * 0：不需要
                               * 1：需要，默认为1
        :param start: 查询任务起始位置，默认为0。
        :param limit: 设定返回任务数量，默认为10。
        :param asc:

                   * 0：降序，默认值
                   * 1：升序
        :param create_time: 任务创建时间，默认为空。
        :type create_time: int
        :param status: 任务状态，默认为空。
                       0:下载成功，1:下载进行中 2:系统错误，3:资源不存在，
                       4:下载超时，5:资源存在但下载失败, 6:存储空间不足,
                       7:目标地址数据已存在, 8:任务取消.
        :type status: int
        :param source_url: 源地址URL，默认为空。
        :param remote_path: 文件保存路径，默认为空。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param expires: 请求失效时间，如果有，则会校验。
        :type expires: int
        :return: Response 对象
        """

        data = {
            'expires': expires,
            'start': start,
            'limit': limit,
            'asc': asc,
            'source_url': source_url,
            'save_path': remote_path,
            'create_time': create_time,
            'status': status,
            'need_task_info': need_task_info,
        }
        return self._request('services/cloud_dl', 'list_task',
                             data=data, **kwargs)