def update(self, CorpNum, MgtKey, cashbill, UserID=None):
        """ 수정
            args
                CorpNum : 팝빌회원 사업자번호
                MgtKey : 원본 현금영수증 문서관리번호
                cashbill : 수정할 현금영수증 object. made with Cashbill(...)
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if cashbill == None:
            raise PopbillException(-99999999, "현금영수증 정보가 입력되지 않았습니다.")

        postData = self._stringtify(cashbill)

        return self._httppost('/Cashbill/' + MgtKey, postData, CorpNum, UserID, "PATCH")