def _on_read_complete(self, data, box):
        """
        完整数据接收完成
        :param data: 原始数据
        :param box: 解析之后的box
        :return:
        """
        msg = dict(
            conn_id=id(self),
            address=self.address,
            data=data,
        )

        # 获取映射的group_id
        group_id = self.factory.app.group_router(box)

        try:
            self.factory.app.parent_output_dict[group_id].put_nowait(msg)
        except:
            logger.error('exc occur. msg: %r', msg, exc_info=True)