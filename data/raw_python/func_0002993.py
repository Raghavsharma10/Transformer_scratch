def getURL(self, CorpNum, UserID, ToGo):
        """ 문자 관련 팝빌 URL
            args
                CorpNum : 팝빌회원 사업자번호
                UserID : 팝빌회원 아이디
                TOGO : BOX (전송내역조회 팝업)
            return
                팝빌 URL
            raise
                PopbillException
        """
        if ToGo == None or ToGo == '':
            raise PopbillException(-99999999, "TOGO값이 입력되지 않았습니다.")

        result = self._httpget('/Message/?TG=' + ToGo, CorpNum, UserID)

        return result.url