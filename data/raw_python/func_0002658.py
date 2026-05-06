def getFaxResultRN(self, CorpNum, RequestNum, UserID=None):
        """ 팩스 전송결과 조회
            args
                CorpNum : 팝빌회원 사업자번호
                RequestNum : 전송요청시 할당한 전송요청번호
                UserID : 팝빌회원 아이디
            return
                팩스전송정보 as list
            raise
                PopbillException
        """

        if RequestNum == None or RequestNum == '':
            raise PopbillException(-99999999, "요청번호가 입력되지 않았습니다.")

        return self._httpget('/FAX/Get/' + RequestNum, CorpNum, UserID)