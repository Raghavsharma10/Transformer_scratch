def sendLMS(self, CorpNum, Sender, Receiver, ReceiverName, Subject, Contents, reserveDT, adsYN=False, UserID=None,
                SenderName=None, RequestNum=None):
        """ 장문 문자메시지 단건 전송
            args
                CorpNum : 팝빌회원 사업자번호
                Sender : 발신번호
                Receiver : 수신번호
                ReceiverName : 수신자명
                Subject : 메시지 제목
                Contents : 메시지 내용(2000Byte 초과시 길이가 조정되어 전송됨)
                reserveDT : 예약전송시간 (형식. yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                SenderName : 발신자명
                RequestNum = 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """

        Messages = []
        Messages.append(MessageReceiver(
            snd=Sender,
            sndnm=SenderName,
            rcv=Receiver,
            rcvnm=ReceiverName,
            msg=Contents,
            sjt=Subject)
        )

        return self.sendMessage("LMS", CorpNum, Sender, '', Subject, Contents, Messages, reserveDT, adsYN, UserID,
                                RequestNum)