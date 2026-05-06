def registDeptUser(self, CorpNum, DeptUserID, DeptUserPWD, UserID=None):
        """ 홈택스 현금영수증 부서사용자 계정 등록
            args
                CorpNum : 팝빌회원 사업자번호
                DeptUserID : 홈택스 부서사용자 계정아이디
                DeptUserPWD : 홈택스 부서사용자 계정비밀번호
                UserID : 팝빌회원 아이디
            return
                처리결과. consist of code and message
            raise
                PopbillException
        """
        if DeptUserID == None or len(DeptUserID) == 0:
            raise PopbillException(-99999999, "홈택스 부서사용자 계정 아이디가 입력되지 않았습니다.")

        if DeptUserPWD == None or len(DeptUserPWD) == 0:
            raise PopbillException(-99999999, "홈택스 부서사용자 계정 비밀번호가 입력되지 않았습니다.")

        req = {}
        req["id"] = DeptUserID
        req["pwd"] = DeptUserPWD

        postData = self._stringtify(req)

        return self._httppost("/HomeTax/Cashbill/DeptUser", postData, CorpNum, UserID)