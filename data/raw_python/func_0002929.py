def getUnitCost(self, CorpNum, ItemCode):
        """ 전자명세서 발행단가 확인.
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
            return
                발행단가 by float
            raise
                PopbillException
        """
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        result = self._httpget('/Statement/' + str(ItemCode) + '?cfg=UNITCOST', CorpNum)
        return float(result.unitCost)