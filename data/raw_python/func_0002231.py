def diff(self, cursor='null', **kwargs):
        """文件增量更新操作查询接口.
        本接口有数秒延迟，但保证返回结果为最终一致.

        :param cursor: 用于标记更新断点。

                       * 首次调用cursor=null；
                       * 非首次调用，使用最后一次调用diff接口的返回结果
                         中的cursor。
        :type cursor: str
        :return: Response 对象
        """

        params = {
            'cursor': cursor,
        }
        return self._request('file', 'diff', extra_params=params, **kwargs)