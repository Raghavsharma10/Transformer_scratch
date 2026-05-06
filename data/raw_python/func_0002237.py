def query_download_tasks(self, task_ids, operate_type=1,
                             expires=None, **kwargs):
        """根据任务ID号，查询离线下载任务信息及进度信息。

        :param task_ids: 要查询的任务ID列表
        :type task_ids: list or tuple
        :param operate_type:
                            * 0：查任务信息
                            * 1：查进度信息，默认为1
        :param expires: 请求失效时间，如果有，则会校验。
        :type expires: int
        :return: Response 对象
        """

        params = {
            'task_ids': ','.join(map(str, task_ids)),
            'op_type': operate_type,
            'expires': expires,
        }
        return self._request('services/cloud_dl', 'query_task',
                             extra_params=params, **kwargs)