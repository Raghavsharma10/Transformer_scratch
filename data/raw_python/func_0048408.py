def do_intersect(self, another_time_range):
        """
        判断与另一时间区间是否有重叠
        :param another_time_range:
        :return: True or False
        """
        if self.begin_dt > another_time_range.end_dt or self.end_dt < \
                another_time_range.begin_dt:
            return False
        else:
            return True