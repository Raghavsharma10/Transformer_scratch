def getUnitCost(self, CorpNum, MsgType, UserID=None):
        """
        전송단가 확인
        :param CorpNum: 팝빌회원 사업자번호
        :param MsgType: 카카오톡 유형
        :param UserID: 팝빌 회원아이디
        :return: unitCost
        """
        if MsgType is None or MsgType == "":
            raise PopbillException(-99999999, "전송유형이 입력되지 않았습니다.")

        result = self._httpget("/KakaoTalk/UnitCost?Type=" + MsgType, CorpNum)
        return float(result.unitCost)