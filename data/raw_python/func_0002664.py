def getPreviewURL(self, CorpNum, ReceiptNum, UserID):
        """ 팩스 발신번호 목록 확인
            args
                CorpNum : 팝빌회원 사업자번호
                UserID : 팝빌회원 아이디
            return
                처리결과. list of SenderNumber
            raise
                PopbillException
        """
        return self._httpget('/FAX/Preview/' + ReceiptNum, CorpNum, UserID).url