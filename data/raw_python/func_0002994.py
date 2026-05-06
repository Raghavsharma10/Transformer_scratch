def getStates(self, Corpnum, reciptNumList, UserID=None):
        """ 전송내역 요약정보 확인
            args
                CorpNum : 팝빌회원 사업자번호
                reciptNumList : 문자전송 접수번호 배열
                UserID : 팝빌회원 아이디
            return
                전송정보 as list
            raise
                PopbillException
        """
        if reciptNumList == None or len(reciptNumList) < 1:
            raise PopbillException(-99999999, "접수번호가 입력되지 않았습니다.")

        postData = self._stringtify(reciptNumList)

        return self._httppost('/Message/States', postData, Corpnum, UserID)