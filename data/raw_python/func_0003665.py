def evaluate_course(self, kcdm, jxbh,
                        r101=1, r102=1, r103=1, r104=1, r105=1, r106=1, r107=1, r108=1, r109=1,
                        r201=3, r202=3, advice=''):
        """
        课程评价, 数值为 1-5, r1 类选项 1 为最好, 5 为最差, r2 类选项程度由深到浅, 3 为最好.

        默认都是最好的选项

        :param kcdm: 课程代码
        :param jxbh: 教学班号
        :param r101: 教学态度认真，课前准备充分
        :param r102: 教授内容充实，要点重点突出
        :param r103: 理论联系实际，反映最新成果
        :param r104: 教学方法灵活，师生互动得当
        :param r105: 运用现代技术，教学手段多样
        :param r106: 注重因材施教，加强能力培养
        :param r107: 严格要求管理，关心爱护学生
        :param r108: 处处为人师表，注重教书育人
        :param r109: 教学综合效果
        :param r201: 课程内容
        :param r202: 课程负担
        :param advice: 其他建议，不能超过120字且不能使用分号,单引号,都好
        :return:
        """
        return self.query(EvaluateCourse(
            kcdm, jxbh,
            r101, r102, r103, r104, r105, r106, r107, r108, r109,
            r201, r202, advice
        ))