def cancel_download_task(self, task_id, expires=None, **kwargs):
        """取消离线下载任务.

        :param task_id: 要取消的任务ID号。
        :type task_id: str
        :param expires: 请求失效时间，如果有，则会校验。
        :type expires: int
        :return: Response 对象
        """

        data = {
            'expires': expires,
            'task_id': task_id,
        }
        return self._request('services/cloud_dl', 'cancle_task',
                             data=data, **kwargs)