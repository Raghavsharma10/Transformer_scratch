def issue(self, CorpNum, MgtKeyType, MgtKey, Memo=None, EmailSubject=None, ForceIssue=False, UserID=None):
        """ 발행
            args
                CorpNum : 회원 사업자 번호
                MgtKeyType : 관리번호 유형 one of ['SELL','BUY','TRUSTEE']
                MgtKey : 파트너 관리번호
                Memo : 처리 메모
                EmailSubject : 발행메일 이메일 제목
                ForceIssue : 지연발행 세금계산서 강제발행 여부.
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

        req = {"forceIssue": ForceIssue}

        if Memo != None and Memo != '':
            req["memo"] = Memo

        if EmailSubject != None and EmailSubject != '':
            req["emailSubject"] = EmailSubject

        postData = self._stringtify(req)

        return self._httppost('/Taxinvoice/' + MgtKeyType + "/" + MgtKey, postData, CorpNum, UserID, "ISSUE")