def getLogs(self, CorpNum, MgtKey):
        """ 문서이력 조회
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 문서관리번호
            return
                문서 이력 목록 as List
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        return self._httpget('/Cashbill/' + MgtKey + '/Logs', CorpNum)