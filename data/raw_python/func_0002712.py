def getTaxinvoice(self, CorpNum, NTSConfirmNum, UserID=None):
        """ 전자세금계산서 상세정보 확인
            args
                CorpNum : 팝빌회원 사업자번호
                NTSConfirmNum : 국세청 승인번호
                UserID : 팝빌회원 아이디
            return
                전자세금계산서 정보객체
            raise
                PopbillException
        """
        if NTSConfirmNum == None or len(NTSConfirmNum) != 24:
            raise PopbillException(-99999999, "국세청승인번호(NTSConfirmNum)가 올바르지 않습니다.")

        return self._httpget('/HomeTax/Taxinvoice/' + NTSConfirmNum, CorpNum, UserID)