def getChargeInfo(self, CorpNum, MsgType, UserID=None):
        """ 과금정보 확인
            args
                CorpNum : 회원 사업자번호
                MsgType : 문자전송 유형
                UserID : 팝빌 회원아이디
            return
                과금정보 객체
            raise
                PopbillException
        """
        if MsgType == None or MsgType == "":
            raise PopbillException(-99999999, "전송유형이 입력되지 않았습니다.")

        return self._httpget('/Message/ChargeInfo?Type=' + MsgType, CorpNum, UserID)