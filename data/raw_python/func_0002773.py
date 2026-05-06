def getUnitCost(self, CorpNum):
        """ 휴폐업조회 단가 확인.
            args
                CorpNum : 팝빌회원 사업자번호
            return
                발행단가 by float
            raise
                PopbillException
        """

        result = self._httpget('/CloseDown/UnitCost', CorpNum)

        return float(result.unitCost)