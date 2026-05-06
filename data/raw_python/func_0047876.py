def get_wrong_answers(self):
        """ provide this method to return only wrong answers
        :return:
        """
        all_answers = self.my_osid_object._my_map['answers']
        wrong_answers = [a for a in all_answers
                         if a['genusTypeId'] == str(WRONG_ANSWER_GENUS_TYPE)]
        return AnswerList(wrong_answers,
                          runtime=self.my_osid_object._runtime,
                          proxy=self.my_osid_object._proxy)