def add_download_task(self, source_url, remote_path,
                          rate_limit=None, timeout=60 * 60,
                          expires=None, callback='', **kwargs):
        """添加离线下载任务，实现单个文件离线下载.

        :param source_url: 源文件的URL。
        :param remote_path: 下载后的文件保存路径。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param rate_limit: 下载限速，默认不限速。
        :type rate_limit: int or long
        :param timeout: 下载超时时间，默认3600秒。
        :param expires: 请求失效时间，如果有，则会校验。
        :type expires: int
        :param callback: 下载完毕后的回调，默认为空。
        :type callback: str
        :return: Response 对象
        """

        data = {
            'source_url': source_url,
            'save_path': remote_path,
            'expires': expires,
            'rate_limit': rate_limit,
            'timeout': timeout,
            'callback': callback,
        }
        return self._request('services/cloud_dl', 'add_task',
                             data=data, **kwargs)