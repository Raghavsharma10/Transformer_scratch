def search_course(self, xqdm, kcdm=None, kcmc=None):
        """
        课程查询

        @structure [{'任课教师': str, '课程名称': str, '教学班号': str, '课程代码': str, '班级容量': int}]

        :param xqdm: 学期代码
        :param kcdm: 课程代码
        :param kcmc: 课程名称
        """
        return self.query(SearchCourse(xqdm, kcdm, kcmc))