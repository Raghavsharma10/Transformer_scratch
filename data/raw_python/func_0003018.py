def getSealURL(self, CorpNum, UserID):
        """ 팝빌 인감 및 첨부문서 등록 URL
            args
                CorpNum : 회원 사업자번호
                UserID  : 회원 팝빌아이디
            return
                30초 보안 토큰을 포함한 url
            raise
                PopbillException
        """
        result = self._httpget('/?TG=SEAL', CorpNum, UserID)
        return result.url