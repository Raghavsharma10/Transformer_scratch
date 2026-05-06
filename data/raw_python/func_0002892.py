def issue(self, CorpNum, MgtKey, Memo=None, UserID=None):
        """ 발행
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 원본 현금영수증 문서관리번호
                Memo : 발행 메모
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """

        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        postData = ""
        req = {}

        if Memo != None or Memo != '':
            req["memo"] = Memo

        postData = self._stringtify(req)

        return self._httppost('/Cashbill/' + MgtKey, postData, CorpNum, UserID, "ISSUE")