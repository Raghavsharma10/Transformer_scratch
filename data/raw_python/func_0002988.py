def sendMMS_Multi(self, CorpNum, Sender, Subject, Contents, Messages, FilePath, reserveDT, adsYN=False, UserID=None,
                      RequestNum=None):
        """ 멀티 문자메시지 다량 전송
            args
                CorpNum : 팝빌회원 사업자번호
                Sender : 발신자번호 (동보전송용)
                Subject : 장문 메시지 제목 (동보전송용)
                Contents : 장문 문자 내용 (동보전송용)
                Messages : 개별전송정보 배열
                FilePath : 전송하고자 하는 파일 경로
                reserveDT : 예약전송시간 (형식. yyyyMMddHHmmss)
                UserID : 팝빌회원 아이디
                RequestNum = 전송요청번호
            return
                접수번호 (receiptNum)
            raise
                PopbillException
        """
        if Messages == None or len(Messages) < 1:
            raise PopbillException(-99999999, "전송할 메시지가 입력되지 않았습니다.")

        req = {}

        if Sender != None or Sender != '':
            req['snd'] = Sender
        if Contents != None or Contents != '':
            req['content'] = Contents
        if Subject != None or Subject != '':
            req['subject'] = Subject
        if reserveDT != None or reserveDT != '':
            req['sndDT'] = reserveDT
        if Messages != None or Messages != '':
            req['msgs'] = Messages
        if RequestNum != None or RequestNum != '':
            req['requestNum'] = RequestNum
        if adsYN:
            req['adsYN'] = True

        postData = self._stringtify(req)

        files = []
        try:
            with open(FilePath, "rb") as F:
                files = [File(fieldName='file',
                              fileName=F.name,
                              fileData=F.read())]
        except IOError:
            raise PopbillException(-99999999, "해당경로에 파일이 없거나 읽을 수 없습니다.")

        result = self._httppost_files('/MMS', postData, files, CorpNum, UserID)

        return result.receiptNum