def multi_restore_recycle_bin(self, fs_ids, **kwargs):
        """批量还原文件或目录（非强一致接口，调用后请sleep1秒 ）.

        :param fs_ids: 所还原的文件或目录在 PCS 的临时唯一标识 ID 的列表。
        :type fs_ids: list or tuple
        :return: Response 对象
        """

        data = {
            'param': json.dumps({
                'list': [{'fs_id': fs_id} for fs_id in fs_ids]
            }),
        }
        return self._request('file', 'restore', data=data, **kwargs)