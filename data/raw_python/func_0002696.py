def sendFTS_same(self, CorpNum, PlusFriendID, Sender, Content, AltContent, AltSendType, SndDT,
                     KakaoMessages, KakaoButtons, AdsYN=False, UserID=None, RequestNum=None):
        """
        친구톡 텍스트 대량 전송
        :param CorpNum: 팝빌회원 사업자번호
        :param PlusFriendID: 플러스친구 아이디
        :param Sender: 발신번호
        :param Content: [동보] 친구톡 내용
        :param AltContent: [동보] 대체문자 내용
        :param AltSendType: 대체문자 유형 [공백-미전송, C-알림톡내용, A-대체문자내용]
        :param SndDT: 예약일시 [작성형식 : yyyyMMddHHmmss]
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
        if AltSendType is not None or AltSendType != '':
            req['altSendType'] = AltSendType
        if Content is not None or Content != '':
            req['content'] = Content
        if AltContent is not None or AltContent != '':
            req['altContent'] = AltContent
        if SndDT is not None or SndDT != '':
            req['sndDT'] = SndDT
        if KakaoMessages:
            req['msgs'] = KakaoMessages
        if KakaoButtons:
            req['btns'] = KakaoButtons
        if AdsYN:
            req['adsYN'] = True
        if RequestNum is not None or RequestNum != '':
            req['requestNum'] = RequestNum

        postData = self._stringtify(req)

        result = self._httppost('/FTS', postData, CorpNum, UserID)

        return result.receiptNum