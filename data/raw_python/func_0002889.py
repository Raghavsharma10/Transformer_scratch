def register(self, CorpNum, cashbill, UserID=None):
        """ 현금영수증 등록
            args
                CorpNum : 팝빌회원 사업자번호
                cashbill : 등록할 현금영수증 object. made with Cashbill(...)
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if cashbill == None:
            raise PopbillException(-99999999, "현금영수증 정보가 입력되지 않았습니다.")

        postData = self._stringtify(cashbill)

        return self._httppost('/Cashbill', postData, CorpNum, UserID)