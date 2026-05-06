def getPartnerURL(self, CorpNum, TOGO):
        """ 팝빌 회원 잔여포인트 확인
            args
                CorpNum : 팝빌회원 사업자번호
                TOGO : "CHRG"
            return
                URL
            raise
                PopbillException
        """
        try:
            return linkhub.getPartnerURL(self._getToken(CorpNum), TOGO)
        except LinkhubException as LE:
            raise PopbillException(LE.code, LE.message)