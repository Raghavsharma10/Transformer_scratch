def getChargeInfo(self, CorpNum, ItemCode, UserID=None):
        """ 과금정보 확인
            args
                CorpNum : 회원 사업자번호
                ItemCode : 전자명세서 종류코드
                UserID : 팝빌 회원아이디
            return
                과금정보 객체
            raise
                PopbillException
        """
        if ItemCode == None or ItemCode == '':
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")

        return self._httpget('/Statement/ChargeInfo/' + ItemCode, CorpNum, UserID)