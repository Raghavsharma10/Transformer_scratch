def sendFax_multi(self, CorpNum, SenderNum, Receiver, FilePath, ReserveDT=None, UserID=None, SenderName=None,
                      adsYN=False, title=None, RequestNum=None):
        """ 팩스 전송
            args
                CorpNum : 팝빌회원 사업자번호
                SenderNum : 발신자 번호 (동보전송용)
                Receiver : 수신자 번호(동보전송용)
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

        if SenderNum == None or SenderNum == "":
            raise PopbillException(-99999999, "발신자 번호가 입력되지 않았습니다.")
        if Receiver == None:
            raise PopbillException(-99999999, "수신자 정보가 입력되지 않았습니다.")
        if not (type(Receiver) is str or type(Receiver) is FaxReceiver or type(Receiver) is list):
            raise PopbillException(-99999999, "'Receiver' argument type error. 'FaxReceiver' or List of 'FaxReceiver'.")
        if FilePath == None:
            raise PopbillException(-99999999, "발신 파일경로가 입력되지 않았습니다.")
        if not (type(FilePath) is str or type(FilePath) is list):
            raise PopbillException(-99999999, "발신 파일은 파일경로 또는 경로목록만 입력 가능합니다.")
        if type(FilePath) is list and (len(FilePath) < 1 or len(FilePath) > 20):
            raise PopbillException(-99999999, "파일은 1개 이상, 20개 까지 전송 가능합니다.")

        req = {"snd": SenderNum, "sndnm": SenderName, "fCnt": 1 if type(FilePath) is str else len(FilePath), "rcvs": [],
               "sndDT": None}

        if (type(Receiver) is str):
            Receiver = FaxReceiver(receiveNum=Receiver)

        if (type(Receiver) is FaxReceiver):
            Receiver = [Receiver]

        if adsYN:
            req['adsYN'] = True

        for r in Receiver:
            req['rcvs'].append({"rcv": r.receiveNum, "rcvnm": r.receiveName})

        if ReserveDT != None:
            req['sndDT'] = ReserveDT

        if title != None:
            req['title'] = title

        if RequestNum != None:
            req['requestNum'] = RequestNum

        postData = self._stringtify(req)

        if (type(FilePath) is str):
            FilePath = [FilePath]

        files = []

        for filePath in FilePath:
            with open(filePath, "rb") as f:
                files.append(File(fieldName='file',
                                  fileName=f.name,
                                  fileData=f.read())
                             )
        result = self._httppost_files('/FAX', postData, files, CorpNum, UserID)

        return result.receiptNum