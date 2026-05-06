def get_answers(self):
        """ override this so only right answers are returned
        :return:
        """
        all_answers = self.my_osid_object._my_map['answers']
        right_answers = [a for a in all_answers
                         if a['genusTypeId'] != str(WRONG_ANSWER_GENUS_TYPE)]
        return AnswerList(right_answers,
                          runtime=self.my_osid_object._runtime,
                          proxy=self.my_osid_object._proxy)