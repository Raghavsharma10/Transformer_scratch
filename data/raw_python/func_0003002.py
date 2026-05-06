def update(self, CorpNum, MgtKeyType, MgtKey, taxinvoice, writeSpecification=False, UserID=None):
        """ 수정
            args
                CorpNum : 회원 사업자 번호
                MgtKeyType : 관리번호 유형 one of ['SELL','BUY','TRUSTEE']
                MgtKey : 파트너 관리번호
                taxinvoice : 수정할 세금계산서 object. Made with Taxinvoice(...)
                writeSpecification : 등록시 거래명세서 동시 작성 여부
                UserID : 팝빌 회원아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKeyType not in self.__MgtKeyTypes:
            raise PopbillException(-99999999, "관리번호 형태가 올바르지 않습니다.")
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if taxinvoice == None:
            raise PopbillException(-99999999, "수정할 세금계산서 정보가 입력되지 않았습니다.")
        if writeSpecification:
            taxinvoice.writeSpecification = True

        postData = self._stringtify(taxinvoice)

        return self._httppost('/Taxinvoice/' + MgtKeyType + '/' + MgtKey, postData, CorpNum, UserID, 'PATCH')