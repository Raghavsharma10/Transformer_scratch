def add_answer(self, vote, rationale):
        """
        Add an answer

        Args:
            vote (int): the option that student voted for
            rationale (str): the reason why the student vote for the option
        """
        self.raw_answers.append({
            VOTE_KEY: vote,
            RATIONALE_KEY: rationale,
        })