def getPopUpURL(self, CorpNum, NTSConfirmNum, UserID=None):
        """ 홈택스 전자세금계산서 보기 팝업 URL
            args
                CorpNum : 팝빌회원 사업자번호
                NTSConfirmNum : 국세청 승인 번호
                UserID : 팝빌회원 아이디
            return
                전자세금계산서 보기 팝업 URL 반환
            raise
                PopbillException
        """

        if NTSConfirmNum == None or len(NTSConfirmNum) != 24:
            raise PopbillException(-99999999, "국세청승인번호(NTSConfirmNum)가 올바르지 않습니다.")

        return self._httpget('/HomeTax/Taxinvoice/' + NTSConfirmNum + '/PopUp', CorpNum, UserID).url