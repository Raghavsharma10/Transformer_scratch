def getUnitCost(self, CorpNum):
        """ 팩스 전송 단가 확인
            args
                CorpNum : 팝빌회원 사업자번호
            return
                전송 단가 by float
            raise
                PopbillException
        """

        result = self._httpget('/FAX/UnitCost', CorpNum)
        return int(result.unitCost)