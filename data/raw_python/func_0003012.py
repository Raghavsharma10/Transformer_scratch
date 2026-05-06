def getFiles(self, CorpNum, MgtKeyType, MgtKey):
        """ 첨부파일 목록 확인
            args
                CorpNum : 회원 사업자 번호
                MgtKeyType : 관리번호 유형 one of ['SELL','BUY','TRUSTEE']
                MgtKey : 파트너 관리번호
            return
                첩부파일 정보 목록 as List
            raise
                PopbillException
        """
        if MgtKeyType not in self.__MgtKeyTypes:
            raise PopbillException(-99999999, "관리번호 형태가 올바르지 않습니다.")
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")

        return self._httpget('/Taxinvoice/' + MgtKeyType + "/" + MgtKey + "/Files", CorpNum)