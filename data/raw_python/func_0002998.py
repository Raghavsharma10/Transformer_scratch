def getCertificateExpireDate(self, CorpNum):
        """ 공인인증서 만료일 확인, 등록여부 확인용도로 활용가능
            args
                CorpNum : 확인할 회원 사업자번호
            return
                등록시 만료일자, 미등록시 해당 PopbillException raise.
            raise
                PopbillException
        """
        result = self._httpget('/Taxinvoice?cfg=CERT', CorpNum)
        return datetime.strptime(result.certificateExpiration, '%Y%m%d%H%M%S')