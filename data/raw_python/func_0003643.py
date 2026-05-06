def sort_hosts(hosts, method='GET', path='/', timeout=(5, 10), **kwargs):
    """
    测试各个地址的速度并返回排名, 当出现错误时消耗时间为 INFINITY = 10000000

    :param method: 请求方法
    :param path: 默认的访问路径
    :param hosts: 进行的主机地址列表, 如 `['http://222.195.8.201/']`
    :param timeout: 超时时间, 可以是一个浮点数或 形如 ``(连接超时, 读取超时)`` 的元祖
    :param kwargs: 其他传递到 ``requests.request`` 的参数
    :return: 形如 ``[(访问耗时, 地址)]`` 的排名数据
    """
    ranks = []

    class HostCheckerThread(Thread):
        def __init__(self, host):
            super(HostCheckerThread, self).__init__()
            self.host = host

        def run(self):
            INFINITY = 10000000
            try:
                url = urllib.parse.urljoin(self.host, path)
                res = requests.request(method, url, timeout=timeout, **kwargs)
                res.raise_for_status()
                cost = res.elapsed.total_seconds() * 1000
            except Exception as e:
                logger.warning('访问出错: %s', e)
                cost = INFINITY
            # http://stackoverflow.com/questions/6319207/are-lists-thread-safe
            ranks.append((cost, self.host))

    threads = [HostCheckerThread(u) for u in hosts]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    ranks.sort()
    return ranks