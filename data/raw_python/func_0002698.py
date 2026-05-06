def cancelReserve(self, CorpNum, ReceiptNum, UserID=None):
        """
        예약전송 취소
        :param CorpNum: 팝빌회원 사업자번호
        :param ReceiptNum: 접수번호
        :param UserID: 팝빌회원 아이디
        :return: code (요청에 대한 상태 응답코드), message (요청에 대한 응답 메시지)
        """
        if ReceiptNum == None or len(ReceiptNum) != 18:
            raise PopbillException(-99999999, "접수번호가 올바르지 않습니다.")

        return self._httpget('/KakaoTalk/' + ReceiptNum + '/Cancel', CorpNum, UserID)