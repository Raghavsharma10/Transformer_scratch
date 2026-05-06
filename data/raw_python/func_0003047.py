def checkIsMember(self, CorpNum):
        """ 회원가입여부 확인
            args
                CorpNum : 회원 사업자번호
            return
                회원가입여부 True/False
            raise
                PopbillException
        """
        if CorpNum == None or CorpNum == '':
            raise PopbillException(-99999999, "사업자번호가 입력되지 않았습니다.")

        return self._httpget('/Join?CorpNum=' + CorpNum + '&LID=' + self.__linkID, None, None)