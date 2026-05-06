def resendFaxRN(self, CorpNum, OrgRequestNum, SenderNum, SenderName, ReceiverNum, ReceiverName, ReserveDT=None,
                    UserID=None, title=None, RequestNum=None):
        """ 팩스 단건 전송
            args
                CorpNum : 팝빌회원 사업자번호
                OrgRequestNum : 원본 팩스 전송시 할당한 전송요청번호
                ReceiptNum : 팩스 접수번호
                SenderNum : 발신자 번호
                SenderName : 발신자명
                ReceiverNum : 수신번호
                ReceiverName : 수신자명
                ReserveDT : 예약시간(형식 yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                title : 팩스제목
                RequestNum : 전송요청시 할당한 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """
        receivers = None

        if ReceiverNum != "" or ReceiverName != "":
            receivers = []
            receivers.append(FaxReceiver(receiveNum=ReceiverNum,
                                         receiveName=ReceiverName)
                             )
        return self.resendFaxRN_multi(CorpNum, OrgRequestNum, SenderNum, SenderName, receivers, ReserveDT,
                                      UserID, title, RequestNum)