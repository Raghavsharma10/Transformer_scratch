def getSenderNumberMgtURL(self, CorpNum, UserID):
        """ 팩스 전송내역 팝업 URL
            args
                CorpNum : 회원 사업자번호
                UserID  : 회원 팝빌아이디
            return
                30초 보안 토큰을 포함한 url
            raise
                PopbillException
        """
        result = self._httpget('/FAX/?TG=SENDER', CorpNum, UserID)
        return result.url