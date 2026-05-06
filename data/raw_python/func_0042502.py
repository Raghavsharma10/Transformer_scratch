def dataReceived(self, data):
        """
        当数据接受到时
        :param data:
        :return:
        """
        self._read_buffer += data

        while self._read_buffer:
            # 因为box后面还是要用的
            box = self.factory.app.box_class()
            ret = box.unpack(self._read_buffer)
            if ret == 0:
                # 说明要继续收
                return
            elif ret > 0:
                # 收好了
                box_data = self._read_buffer[:ret]
                self._read_buffer = self._read_buffer[ret:]
                safe_call(self._on_read_complete, box_data, box)
                continue
            else:
                # 数据已经混乱了，全部丢弃
                logger.error('buffer invalid. ret: %d, read_buffer: %r', ret, self._read_buffer)
                self._read_buffer = ''
                return