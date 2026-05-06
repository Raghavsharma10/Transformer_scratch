def learn(self, msg, learnas):
        """Learn message as spam/ham or forget"""
        if not isinstance(learnas, types.StringTypes):
            raise SpamCError('The learnas option is invalid')
        if learnas.lower() == 'forget':
            resp = self.tell(msg, 'forget')
        else:
            resp = self.tell(msg, 'learn', learnas)
        return resp