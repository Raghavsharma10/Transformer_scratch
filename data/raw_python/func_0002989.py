def sendMessage(self, MsgType, CorpNum, Sender, SenderName, Subject, Contents, Messages, reserveDT, adsYN=False,
                    UserID=None, RequestNum=None):
        """ 문자 메시지 전송
            args
                MsgType : 문자 전송 유형(단문:SMS, 장문:LMS, 단/장문:XMS)
                CorpNum : 팝빌회원 사업자번호
                Sender : 발신자번호 (동보전송용)
                Subject : 장문 메시지 제목 (동보전송용)
                Contents : 장문 문자 내용 (동보전송용)
                Messages : 개별전송정보 배열
                reserveDT : 예약전송시간 (형식. yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                RequestNum : 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """
        if MsgType == None or MsgType == '':
            raise PopbillException(-99999999, "문자 전송 유형이 입력되지 않았습니다.")

        if Messages == None or len(Messages) < 1:
            raise PopbillException(-99999999, "전송할 메시지가 입력되지 않았습니다.")

        req = {}

        if Sender != None or Sender != '':
            req['snd'] = Sender
        if SenderName != None or SenderName != '':
            req['sndnm'] = SenderName
        if Contents != None or Contents != '':
            req['content'] = Contents
        if Subject != None or Subject != '':
            req['subject'] = Subject
        if reserveDT != None or reserveDT != '':
            req['sndDT'] = reserveDT
        if Messages != None or Messages != '':
            req['msgs'] = Messages
        if RequestNum != None or RequestNum != '':
            req['requestnum'] = RequestNum
        if adsYN:
            req['adsYN'] = True

        postData = self._stringtify(req)

        result = self._httppost('/' + MsgType, postData, CorpNum, UserID)

        return result.receiptNum