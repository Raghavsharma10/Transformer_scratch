def send(self, request, **kwargs):
        """
        所有接口用来发送请求的方法, 只是 :meth:`requests.sessions.Session.send` 的一个钩子方法, 用来处理请求前后的工作
        """
        response = super(BaseSession, self).send(request, **kwargs)
        if ENV['RAISE_FOR_STATUS']:
            response.raise_for_status()

        parsed = parse.urlparse(response.url)
        if parsed.netloc == parse.urlparse(self.host).netloc:
            response.encoding = ENV['SITE_ENCODING']
            # 快速判断响应 IP 是否被封, 那个警告响应内容长度为 327 或 328, 保留一点余量确保准确
            min_length, max_length, pattern = ENV['IP_BANNED_RESPONSE']
            if min_length <= len(response.content) <= max_length and pattern.search(response.text):
                msg = '当前 IP 已被锁定, 如果可以请尝试切换教务系统地址, 否则请在更换网络环境或等待解封后重试!'
                raise IPBanned(msg)

        self.histories.append(response)
        logger.debug(report_response(response, redirection=kwargs.get('allow_redirects')))
        return response