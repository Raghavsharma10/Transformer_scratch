def restore_recycle_bin(self, fs_id, **kwargs):
        """还原单个文件或目录（非强一致接口，调用后请sleep 1秒读取）.

        :param fs_id: 所还原的文件或目录在PCS的临时唯一标识ID。
        :type fs_id: str
        :return: Response 对象
        """

        data = {
            'fs_id': fs_id,
        }
        return self._request('file', 'restore', data=data, **kwargs)