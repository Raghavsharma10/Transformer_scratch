def check_courses(self, kcdms):
        """
        检查课程是否被选

        @structure [bool]

        :param kcdms: 课程代码列表
        :return: 与课程代码列表长度一致的布尔值列表, 已为True,未选为False
        """
        selected_courses = self.get_selected_courses()
        selected_kcdms = {course['课程代码'] for course in selected_courses}
        result = [True if kcdm in selected_kcdms else False for kcdm in kcdms]
        return result