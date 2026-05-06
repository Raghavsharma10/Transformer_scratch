def sendFax(self, CorpNum, SenderNum, ReceiverNum, ReceiverName, FilePath, ReserveDT=None, UserID=None,
                SenderName=None, adsYN=False, title=None, RequestNum=None):
        """ 팩스 단건 전송
            args
                CorpNum : 팝빌회원 사업자번호
                SenderNum : 발신자 번호
                ReceiverNum : 수신자 번호
                ReceiverName : 수신자 명
                FilePath : 발신 파일경로
                ReserveDT : 예약시간(형식 yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                SenderName : 발신자명 (동보전송용)
                adsYN : 광고팩스 여부
                title : 팩스제목
                RequestNum : 전송요청시 할당한 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """
        receivers = []
        receivers.append(FaxReceiver(receiveNum=ReceiverNum,
                                     receiveName=ReceiverName)
                         )

        return self.sendFax_multi(CorpNum, SenderNum, receivers, FilePath, ReserveDT, UserID, SenderName, adsYN, title,
                                  RequestNum)