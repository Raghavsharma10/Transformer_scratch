def get_mv_detail(self, mvid):
        """Get mv detail
        :param mvid: mv id
        :return:
        """
        url = uri + '/mv/detail?id=' + str(mvid)
        return self.request('GET', url)