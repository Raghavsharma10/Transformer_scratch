def change_course(self, select_courses=None, delete_courses=None):
        """
        修改个人的课程

        @structure [{'费用': float, '教学班号': str, '课程名称': str, '课程代码': str, '学分': float, '课程类型': str}]

        :param select_courses: 形如 ``[{'kcdm': '9900039X', 'jxbhs': {'0001', '0002'}}]`` 的课程代码与教学班号列表,
          jxbhs 可以为空代表选择所有可选班级
        :param delete_courses: 需要删除的课程代码集合, 如 ``{'0200011B'}``
        :return: 选课结果, 返回选中的课程教学班列表, 结构与 ``get_selected_courses`` 一致
        """
        # 框架重构后调整接口的调用
        t = self.get_system_status()
        if t['当前轮数'] is None:
            raise ValueError('当前为 %s,选课系统尚未开启', t['当前学期'])
        if not (select_courses or delete_courses):
            raise ValueError('select_courses, delete_courses 参数不能都为空!')
        # 参数处理
        select_courses = select_courses or []
        delete_courses = {l.upper() for l in (delete_courses or [])}

        selected_courses = self.get_selected_courses()
        selected_kcdms = {course['课程代码'] for course in selected_courses}

        # 尝试删除没有被选中的课程会出错
        unselected = delete_courses.difference(selected_kcdms)
        if unselected:
            msg = '无法删除没有被选的课程 {}'.format(unselected)
            logger.warning(msg)

        # 要提交的 kcdm 数据
        kcdms_data = []
        # 要提交的 jxbh 数据
        jxbhs_data = []

        # 必须添加已选课程, 同时去掉要删除的课程
        for course in selected_courses:
            if course['课程代码'] not in delete_courses:
                kcdms_data.append(course['课程代码'])
                jxbhs_data.append(course['教学班号'])

        # 选课
        for kv in select_courses:
            kcdm = kv['kcdm'].upper()
            jxbhs = set(kv['jxbhs']) if kv.get('jxbhs') else set()

            teaching_classes = self.get_course_classes(kcdm)
            if not teaching_classes:
                logger.warning('课程[%s]没有可选班级', kcdm)
                continue

            # 反正是统一提交, 不需要判断是否已满
            optional_jxbhs = {c['教学班号'] for c in teaching_classes['可选班级']}
            if jxbhs:
                wrong_jxbhs = jxbhs.difference(optional_jxbhs)
                if wrong_jxbhs:
                    msg = '课程[{}]{}没有教学班号{}'.format(kcdm, teaching_classes['课程名称'], wrong_jxbhs)
                    logger.warning(msg)
                jxbhs = jxbhs.intersection(optional_jxbhs)
            else:
                jxbhs = optional_jxbhs

            for jxbh in jxbhs:
                kcdms_data.append(kcdm)
                jxbhs_data.append(jxbh)

        return self.query(ChangeCourse(self.session.account, select_courses, delete_courses))