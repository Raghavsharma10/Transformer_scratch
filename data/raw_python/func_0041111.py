def round_to(self, dt, hour, minute, second, mode="floor"):
        """Round the given datetime to specified hour, minute and second.

        :param mode: 'floor' or 'ceiling'

        **中文文档**

        将给定时间对齐到最近的一个指定了小时, 分钟, 秒的时间上。
        """
        mode = mode.lower()

        new_dt = datetime(dt.year, dt.month, dt.day, hour, minute, second)
        if mode == "floor":
            if new_dt <= dt:
                return new_dt
            else:
                return rolex.add_days(new_dt, -1)
        elif mode == "ceiling":
            if new_dt >= dt:
                return new_dt
            else:
                return rolex.add_days(new_dt, 1)
        else:
            raise ValueError("'mode' has to be 'floor' or 'ceiling'!")