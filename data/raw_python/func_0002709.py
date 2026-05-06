def getJobState(self, CorpNum, JobID, UserID=None):
        """ 수집 상태 확인
            args
                CorpNum : 팝빌회원 사업자번호
                JobID : 작업아이디
                UserID : 팝빌회원 아이디
            return
                수집 상태 정보
            raise
                PopbillException
        """
        if JobID == None or len(JobID) != 18:
            raise PopbillException(-99999999, "작업아이디(jobID)가 올바르지 않습니다.")

        return self._httpget('/HomeTax/Taxinvoice/' + JobID + '/State', CorpNum, UserID)