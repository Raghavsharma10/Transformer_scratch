def registRequest(self, CorpNum, taxinvoice, memo=None, UserID=None):
        """ 즉시 요청
            args
                CorpNum : 팝빌회원 사업자번호
                taxinvoice : 세금계산서 객체
                memo : 메모
                UsreID : 팝빌회원 아이디
            return
                검색결과 정보
            raise
                PopbillException
        """

        if memo != None and memo != '':
            taxinvoice.memo = memo

        postData = self._stringtify(taxinvoice)

        return self._httppost('/Taxinvoice', postData, CorpNum, UserID, "REQUEST")