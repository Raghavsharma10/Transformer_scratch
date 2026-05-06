def sendATS_same(self, CorpNum, TemplateCode, Sender, Content, AltContent, AltSendType, SndDT, KakaoMessages,
                     UserID=None, RequestNum=None, ButtonList=None):
        """
       알림톡 대량 전송
       :param CorpNum: 팝빌회원 사업자번호
       :param TemplateCode: 템플릿코드
       :param Sender: 발신번호
       :param Content: [동보] 알림톡 내용
       :param AltContent: [동보] 대체문자 내용
       :param AltSendType: 대체문자 유형 [공백-미전송, C-알림톡내용, A-대체문자내용]
       :param SndDT: 예약일시 [작성형식 : yyyyMMddHHmmss]
       :param KakaoMessages: 알림톡 내용 (배열)
       :param UserID: 팝빌회원 아이디
       :param RequestNum : 요청번호
       :return: receiptNum (접수번호)
       """
        if TemplateCode is None or TemplateCode == '':
            raise PopbillException(-99999999, "알림톡 템플릿코드가 입력되지 않았습니다.")
        if Sender is None or Sender == '':
            raise PopbillException(-99999999, "발신번호가 입력되지 않았습니다.")

        req = {}

        if TemplateCode is not None or TemplateCode != '':
            req['templateCode'] = TemplateCode
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
        if ButtonList is not None:
            req['btns'] = ButtonList
        if RequestNum is not None or RequestNum != '':
            req['requestnum'] = RequestNum

        postData = self._stringtify(req)

        result = self._httppost('/ATS', postData, CorpNum, UserID)

        return result.receiptNum