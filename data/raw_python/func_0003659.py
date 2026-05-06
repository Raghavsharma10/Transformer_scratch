def get_class_students(self, xqdm, kcdm, jxbh):
        """
        教学班查询, 查询指定教学班的所有学生

        @structure {'学期': str, '班级名称': str, '学生': [{'姓名': str, '学号': int}]}

        :param xqdm: 学期代码
        :param kcdm: 课程代码
        :param jxbh: 教学班号
        """
        return self.query(GetClassStudents(xqdm, kcdm, jxbh))