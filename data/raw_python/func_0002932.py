def register(self, CorpNum, statement, UserID=None):
        """ 임시저장
            args
                CorpNum : 팝빌회원 사업자번호
                statement : 등록할 전자명세서 object. made with Statement(...)
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if statement == None:
            raise PopbillException(-99999999, "등록할 전자명세서 정보가 입력되지 않았습니다.")

        postData = self._stringtify(statement)

        return self._httppost('/Statement', postData, CorpNum, UserID)