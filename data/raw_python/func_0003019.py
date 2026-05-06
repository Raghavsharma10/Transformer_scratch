def getTaxCertURL(self, CorpNum, UserID):
        """ 공인인증서 등록 URL
            args
                CorpNum : 회원 사업자번호
                UserID  : 회원 팝빌아이디
            return
                30초 보안 토큰을 포함한 url
            raise
                PopbillException
        """
        result = self._httpget('/?TG=CERT', CorpNum, UserID)
        return result.url