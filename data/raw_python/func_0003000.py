def register(self, CorpNum, taxinvoice, writeSpecification=False, UserID=None):
        """ 임시저장
            args
                CorpNum : 회원 사업자 번호
                taxinvoice : 등록할 세금계산서 object. Made with Taxinvoice(...)
                writeSpecification : 등록시 거래명세서 동시 작성 여부
                UserID : 팝빌 회원아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if taxinvoice == None:
            raise PopbillException(-99999999, "등록할 세금계산서 정보가 입력되지 않았습니다.")
        if writeSpecification:
            taxinvoice.writeSpecification = True

        postData = self._stringtify(taxinvoice)

        return self._httppost('/Taxinvoice', postData, CorpNum, UserID)