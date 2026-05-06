def __get_match_result(self, ret, ret2):
        """
        Getting match result
        """
        if self.another_compare == "__MATCH_AND__":
            return ret and ret2
        elif self.another_compare == "__MATCH_OR__":
            return ret or ret2
        return ret