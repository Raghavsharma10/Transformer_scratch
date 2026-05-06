def getInfos(self, CorpNum, MgtKeyList):
        """ 상태정보 다량 확인, 최대 1000건
            args
                CorpNum : 회원 사업자 번호
                MgtKeyList : 문서관리번호 목록
            return
                상태정보 목록 as List
            raise
                PopbillException
        """
        if MgtKeyList == None or len(MgtKeyList) < 1:
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        postData = self._stringtify(MgtKeyList)

        return self._httppost('/Cashbill/States', postData, CorpNum)