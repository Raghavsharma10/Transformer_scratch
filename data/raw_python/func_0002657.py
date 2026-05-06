def getFaxResult(self, CorpNum, ReceiptNum, UserID=None):
        """ 팩스 전송결과 조회
            args
                CorpNum : 팝빌회원 사업자번호
                ReceiptNum : 전송요청시 발급받은 접수번호
                UserID : 팝빌회원 아이디
            return
                팩스전송정보 as list
            raise
                PopbillException
        """

        if ReceiptNum == None or len(ReceiptNum) != 18:
            raise PopbillException(-99999999, "접수번호가 올바르지 않습니다.")

        return self._httpget('/FAX/' + ReceiptNum, CorpNum, UserID)