def get_teaching_plan(self, xqdm, kclx='b', zydm=''):
        """
        专业教学计划查询, 可以查询全校公选课, 此时可以不填 `zydm`

        @structure [{'开课单位': str, '学时': int, '课程名称': str, '课程代码': str, '学分': float}]

        :param xqdm: 学期代码
        :param kclx: 课程类型参数,只有两个值 b:专业必修课, x:全校公选课
        :param zydm: 专业代码, 可以从 :meth:`models.StudentSession.get_code` 获得
        """
        return self.query(GetTeachingPlan(xqdm, kclx, zydm))