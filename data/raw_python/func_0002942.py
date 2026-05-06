def attachFile(self, CorpNum, ItemCode, MgtKey, FilePath, UserID=None):
        """ 파일 첨부
            args
                CorpNum : 팝빌회원 사업자번호
                ItemCode : 명세서 종류 코드
                    [121 - 거래명세서], [122 - 청구서], [123 - 견적서],
                    [124 - 발주서], [125 - 입금표], [126 - 영수증]
                MgtKey : 파트너 문서관리번호
                FilePath : 첨부파일의 경로
                UserID : 팝빌 회원아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if MgtKey == None or MgtKey == "":
            raise PopbillException(-99999999, "관리번호가 입력되지 않았습니다.")
        if FilePath == None or FilePath == "":
            raise PopbillException(-99999999, "파일경로가 입력되지 않았습니다.")
        if ItemCode == None or ItemCode == "":
            raise PopbillException(-99999999, "명세서 종류 코드가 입력되지 않았습니다.")
        files = []

        try:
            with open(FilePath, "rb") as F:
                files = [File(fieldName='Filedata',
                              fileName=F.name,
                              fileData=F.read())]
        except IOError:
            raise PopbillException(-99999999, "해당경로에 파일이 없거나 읽을 수 없습니다.")

        return self._httppost_files('/Statement/' + str(ItemCode) + '/' + MgtKey + '/Files', None, files, CorpNum,
                                    UserID)