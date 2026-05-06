def rank(self, score):
        '''Return the 0-based index (rank) of ``score``. If the score is not
available it returns a negative integer which absolute score is the
left most closest index with score less than *score*.'''
        node = self.__head
        rank = 0
        for i in range(self.__level-1, -1, -1):
            while node.next[i] and node.next[i].score <= score:
                rank += node.width[i]
                node = node.next[i]
        if node.score == score:
            return rank - 1
        else:
            return -1 - rank