def checkCorpNums(self, MemberCorpNum, CorpNumList):
        """ 휴폐업조회 대량 확인, 최대 1000건
            args
                MemberCorpNum : 팝빌회원 사업자번호
                CorpNumList : 조회할 사업자번호 배열
            return
                휴폐업정보 Object as List
            raise
                PopbillException
        """
        if CorpNumList == None or len(CorpNumList) < 1:
            raise PopbillException(-99999999,"조죄할 사업자번호 목록이 입력되지 않았습니다.")

        postData = self._stringtify(CorpNumList)

        return self._httppost('/CloseDown',postData,MemberCorpNum)