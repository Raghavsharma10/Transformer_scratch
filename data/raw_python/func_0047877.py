def get_wrong_answer_ids(self):
        """provide this method to return only wrong answer ids"""
        id_list = []
        for answer in self.get_wrong_answers():
            id_list.append(answer.get_id())
        return IdList(id_list)