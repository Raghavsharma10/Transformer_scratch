def sendSMS_multi(self, CorpNum, Sender, Contents, Messages, reserveDT, adsYN=False, UserID=None, RequestNum=None):
        """ 단문 문자메시지 다량전송
            args
                CorpNum : 팝빌회원 사업자번호
                Sender : 발신자번호 (동보전송용)
                Contents : 문자 내용 (동보전송용)
                Messages : 개별전송정보 배열
                reserveDT : 예약전송시간 (형식. yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                RequestNum : 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """

        return self.sendMessage("SMS", CorpNum, Sender, '', '', Contents, Messages, reserveDT, adsYN, UserID,
                                RequestNum)