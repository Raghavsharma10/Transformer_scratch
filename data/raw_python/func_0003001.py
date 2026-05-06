def registIssue(self, CorpNum, taxinvoice, writeSpecification=False, forceIssue=False, dealInvoiceMgtKey=None,
                    memo=None, emailSubject=None, UserID=None):
        """ 즉시 발행
            args
                CorpNum : 팝빌회원 사업자번호
                taxinvoice : 세금계산서 객체
                writeSpecification : 거래명세서 동시작성 여부
                forceIssue : 지연발행 강제여부
                dealInvoiceMgtKey : 거래명세서 문서관리번호
                memo : 메모
                emailSubject : 메일제목, 미기재시 기본제목으로 전송
                UsreID : 팝빌회원 아이디
            return
                검색결과 정보
            raise
                PopbillException
        """
        if writeSpecification:
            taxinvoice.writeSpecification = True

        if forceIssue:
            taxinvoice.forceIssue = True

        if dealInvoiceMgtKey != None and dealInvoiceMgtKey != '':
            taxinvoice.dealInvoiceMgtKey = dealInvoiceMgtKey

        if memo != None and memo != '':
            taxinvoice.memo = memo

        if emailSubject != None and emailSubject != '':
            taxinvoice.emailSubject = emailSubject

        postData = self._stringtify(taxinvoice)

        return self._httppost('/Taxinvoice', postData, CorpNum, UserID, "ISSUE")