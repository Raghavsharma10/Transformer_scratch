def getInfo(self, CorpNum, MgtKeyType, MgtKey):
        """ 상태정보 확인
            args
                CorpNum : 회원 사업자 번호
                MgtKeyType : 관리번호 유형 one of ['SELL','BUY','TRUSTEE']
                MgtKey : 파트너 관리번호
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKeyType not in self.__MgtKeyTypes:
            raise PopbillException(-99999999, "관리번호 형태가 올바르지 않습니다.")
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        return self._httpget('/Taxinvoice/' + MgtKeyType + '/' + MgtKey, CorpNum)