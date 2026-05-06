def resendFaxRN_multi(self, CorpNum, OrgRequestNum, SenderNum, SenderName, Receiver, ReserveDT=None, UserID=None,
                          title=None, RequestNum=None):
        """ 팩스 전송
            args
                CorpNum : 팝빌회원 사업자번호
                OrgRequestNum : 원본 팩스 전송시 할당한 전송요청번호
                SenderNum : 발신자 번호
                SenderName : 발신자명
                Receiver : 수신자정보 배열
                ReserveDT : 예약시간(형식 yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                title : 팩스제목
                RequestNum : 전송요청시 할당한 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """

        req = {}

        if not OrgRequestNum:
            raise PopbillException(-99999999, "원본 팩스 요청번호가 입력되지 않았습니다")

        if SenderNum != "":
            req['snd'] = SenderNum

        if SenderName != "":
            req['sndnm'] = SenderName

        if ReserveDT != None:
            req['sndDT'] = ReserveDT

        if title != None:
            req['title'] = title

        if RequestNum != None:
            req['requestNum'] = RequestNum

        if Receiver != None:
            req['rcvs'] = []
            if (type(Receiver) is str):
                Receiver = FaxReceiver(receiveNum=Receiver)
            if (type(Receiver) is FaxReceiver):
                Receiver = [Receiver]
            for r in Receiver:
                req['rcvs'].append({"rcv": r.receiveNum, "rcvnm": r.receiveName})

        postData = self._stringtify(req)

        return self._httppost('/FAX/Resend/' + OrgRequestNum, postData, CorpNum, UserID).receiptNum