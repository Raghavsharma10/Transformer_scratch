def sendFMS_same(self, CorpNum, PlusFriendID, Sender, Content, AltContent, AltSendType, SndDT, FilePath, ImageURL,
                     KakaoMessages, KakaoButtons, AdsYN=False, UserID=None, RequestNum=None):
        """
        친구톡 이미지 대량 전송
        :param CorpNum: 팝빌회원 사업자번호
        :param PlusFriendID: 플러스친구 아이디
        :param Sender: 발신번호
        :param Content: [동보] 친구톡 내용
        :param AltContent: [동보] 대체문자 내용
        :param AltSendType: 대체문자 유형 [공백-미전송, C-알림톡내용, A-대체문자내용]
        :param SndDT: 예약일시 [작성형식 : yyyyMMddHHmmss]
        :param FilePath: 파일경로
        :param ImageURL: 이미지URL
        :param KakaoMessages: 친구톡 내용 (배열)
        :param KakaoButtons: 버튼 목록 (최대 5개)
        :param AdsYN: 광고 전송여부
        :param UserID: 팝빌회원 아이디
        :param RequestNum : 요청번호
        :return: receiptNum (접수번호)
        """
        if PlusFriendID is None or PlusFriendID == '':
            raise PopbillException(-99999999, "플러스친구 아이디가 입력되지 않았습니다.")
        if Sender is None or Sender == '':
            raise PopbillException(-99999999, "발신번호가 입력되지 않았습니다.")

        req = {}
        if PlusFriendID is not None or PlusFriendID != '':
            req['plusFriendID'] = PlusFriendID
        if Sender is not None or Sender != '':
            req['snd'] = Sender
        if Content is not None or Content != '':
            req['content'] = Content
        if AltContent is not None or AltContent != '':
            req['altContent'] = AltContent
        if AltSendType is not None or AltSendType != '':
            req['altSendType'] = AltSendType
        if SndDT is not None or SndDT != '':
            req['sndDT'] = SndDT
        if KakaoMessages is not None or KakaoMessages != '':
            req['msgs'] = KakaoMessages
        if ImageURL is not None or ImageURL != '':
            req['imageURL'] = ImageURL
        if KakaoButtons:
            req['btns'] = KakaoButtons
        if AdsYN:
            req['adsYN'] = True
        if RequestNum is not None or RequestNum != '':
            req['requestNum'] = RequestNum

        postData = self._stringtify(req)

        files = []
        try:
            with open(FilePath, "rb") as F:
                files = [File(fieldName='file',
                              fileName=F.name,
                              fileData=F.read())]
        except IOError:
            raise PopbillException(-99999999, "해당경로에 파일이 없거나 읽을 수 없습니다.")

        result = self._httppost_files('/FMS', postData, files, CorpNum, UserID)

        return result.receiptNum