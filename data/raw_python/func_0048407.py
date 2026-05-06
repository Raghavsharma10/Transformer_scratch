def lie_between(self, target_time_range):
        """
        判断是否落在目标时间区间内
        :param target_time_range: 目标时间区间
        :return: True or False
        """
        if self.begin_dt >= target_time_range.begin_dt and self.end_dt <= \
                target_time_range.end_dt:
            return True
        else:
            return False