def most_similar(self, keyword, num):
        """
        input: keyword term of top n
        output: keyword result in json formmat
        """
        try:
            result = self.model.most_similar(keyword, topn = num) # most_similar return a list
            return {'key':keyword, 'value':result, 'similarity':1}
        except KeyError as e:
            kemKeyword = self.kemNgram.find(keyword)
            if kemKeyword:
                result = self.model.most_similar(kemKeyword, topn = num)
                return {'key':kemKeyword, 'value':result, 'similarity':self.kemNgram.compare(kemKeyword, keyword)}
            return {'key':keyword, 'value':[], 'similarity':0}