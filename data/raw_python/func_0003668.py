def get_selectable_courses(self, kcdms=None, pool_size=5, dump_result=True, filename='可选课程.json', encoding='utf-8'):
        """
        获取所有能够选上的课程的课程班级, 注意这个方法遍历所给出的课程和它们的可选班级, 当选中人数大于等于课程容量时表示不可选.

        由于请求非常耗时且一般情况下用不到, 因此默认推荐在第一轮选课结束后到第三轮选课结束之前的时间段使用, 如果你仍然坚持使用, 你将会得到一个警告.

        @structure [{'可选班级': [{'起止周': str, '考核类型': str, '教学班附加信息': str, '课程容量': int, '选中人数': int,
         '教学班号': str, '禁选专业': str, '教师': [str], '校区': str, '优选范围': [str], '开课时间,开课地点': [str]}],
        '课程代码': str, '课程名称': str}]

        :param kcdms: 课程代码列表, 默认为所有可选课程的课程代码
        :param dump_result: 是否保存结果到本地
        :param filename: 保存的文件路径
        :param encoding: 文件编码
        """
        now = time.time()
        t = self.get_system_status()
        if not (t['选课计划'][0][1] < now < t['选课计划'][2][1]):
            logger.warning('只推荐在第一轮选课结束到第三轮选课结束之间的时间段使用本接口!')

        def iter_kcdms():
            for l in self.get_optional_courses():
                yield l['课程代码']

        kcdms = kcdms or iter_kcdms()

        def target(kcdm):
            course_classes = self.get_course_classes(kcdm)
            if not course_classes:
                return
            course_classes['可选班级'] = [c for c in course_classes['可选班级'] if c['课程容量'] > c['选中人数']]
            if len(course_classes['可选班级']) > 0:
                return course_classes

        # Python 2.7 不支持 with 语法
        pool = Pool(pool_size)
        result = list(filter(None, pool.map(target, kcdms)))
        pool.close()
        pool.join()

        if dump_result:
            json_str = json.dumps(result, ensure_ascii=False, indent=4, sort_keys=True)
            with open(filename, 'wb') as fp:
                fp.write(json_str.encode(encoding))
            logger.info('可选课程结果导出到了:%s', filename)
        return result