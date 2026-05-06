def get_answer_ids(self):
        """ override this so only right answer ids are returned
        :return:
        """
        id_list = []
        for answer in self.get_answers():
            id_list.append(answer.get_id())
        return IdList(id_list)