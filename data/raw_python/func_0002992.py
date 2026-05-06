def cancelReserveRN(self, CorpNum, RequestNum, UserID=None):
        """ 문자 예약전송 취소
            args
                CorpNum : 팝빌회원 사업자번호
                RequestNum : 전송요청시 할당한 전송요청번호
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if RequestNum == None or RequestNum == '':
            raise PopbillException(-99999999, "요청번호가 입력되지 않았습니다.")

        return self._httpget('/Message/Cancel/' + RequestNum, CorpNum, UserID)