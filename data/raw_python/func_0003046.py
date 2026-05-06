def getChargeURL(self, CorpNum, UserID):
        """ 팝빌 연동회원 포인트 충전 URL
            args
                CorpNum : 회원 사업자번호
                UserID  : 회원 팝빌아이디
            return
                30초 보안 토큰을 포함한 url
            raise
                PopbillException
        """
        result = self._httpget('/?TG=CHRG', CorpNum, UserID)
        return result.url