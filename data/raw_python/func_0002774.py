def checkCorpNum(self, MemberCorpNum, CheckCorpNum):
        """ 휴폐업조회 - 단건
            args
                MemberCorpNum : 팝빌회원 사업자번호
                CorpNum : 조회할 사업자번호
                MgtKey : 문서관리번호
            return
                휴폐업정보 object
            raise
                PopbillException
        """

        if MemberCorpNum == None or MemberCorpNum == "" :
            raise PopbillException(-99999999,"팝빌회원 사업자번호가 입력되지 않았습니다.")

        if CheckCorpNum == None or CheckCorpNum == "" :
            raise PopbillException(-99999999,"조회할 사업자번호가 입력되지 않았습니다.")

        return self._httpget('/CloseDown?CN=' +CheckCorpNum, MemberCorpNum)