def getDetailInfo(self, CorpNum, MgtKey):
        """ 상세정보 조회
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 문서관리번호
            return
                문서 상세정보 object
            raise
                PopbillException
        """

        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        return self._httpget('/Cashbill/' + MgtKey + '?Detail', CorpNum)