def getBalance(self, CorpNum):
        """ 팝빌 회원 잔여포인트 확인
            args
                CorpNum : 확인하고자 하는 회원 사업자번호
            return
                잔여포인트 by float
            raise
                PopbillException
        """
        try:
            return linkhub.getBalance(self._getToken(CorpNum))
        except LinkhubException as LE:
            raise PopbillException(LE.code, LE.message)