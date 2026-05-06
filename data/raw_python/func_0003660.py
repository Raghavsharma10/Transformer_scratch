def get_class_info(self, xqdm, kcdm, jxbh):
        """
        获取教学班详情, 包括上课时间地点, 考查方式, 老师, 选中人数, 课程容量等等信息

        @structure {'校区': str,'开课单位': str,'考核类型': str,'课程类型': str,'课程名称': str,'教学班号': str,'起止周': str,
        '时间地点': str,'学分': float,'性别限制': str,'优选范围': str,'禁选范围': str,'选中人数': int,'备注': str}

        :param xqdm: 学期代码
        :param kcdm: 课程代码
        :param jxbh: 教学班号
        """
        return self.query(GetClassInfo(xqdm, kcdm, jxbh))