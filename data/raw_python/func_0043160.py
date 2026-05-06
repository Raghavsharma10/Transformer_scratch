def missing(self):
        """
        THERE IS PLENTY OF OPPORTUNITY TO SIMPLIFY missing EXPRESSIONS
        OVERRIDE THIS METHOD TO SIMPLIFY
        :return:
        """
        if self.type == BOOLEAN:
            Log.error("programmer error")
        return self.lang[MissingOp(self)]