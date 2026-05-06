def add_report_data(list_all=[], module_name="TestModule", **kwargs):
        ''' add report data to a list
            @param list_all: a list which save the report data
            @param module_name: test set name or test module name
            @param kwargs: such as
                case_name:   testcase name
                status:      test result, Pass or Fail
                resp_tester: responsible tester who write this case
                tester:      tester who execute the test
                start_at:    tester run this case at time 
                end_at:      tester stop this case at time
        '''
        start_at = kwargs.get("start_at")        
        case_name = kwargs.get("case_name","TestCase")
        raw_case_name = kwargs.get("raw_case_name","TestCase")
                
        exec_date_time = time.localtime(start_at)
        execdate = time.strftime("%Y-%m-%d",exec_date_time) 
        exectime = time.strftime("%H:%M:%S",exec_date_time)
        
        _case_report = {
                'resp_tester': kwargs.get("resp_tester","administrator"),
                'tester': kwargs.get("tester","administrator"),
                'case_name': case_name,
                'raw_case_name': raw_case_name,
                'status': kwargs.get("status","Pass"),
                'exec_date': execdate,
                'exec_time': exectime,
                'start_at': start_at,
                'end_at': kwargs.get("end_at"),
            }
                
        for module in list_all:
            if module_name != module["Name"]:
                continue
            
            for case in module["TestCases"]:
                if raw_case_name == case["raw_case_name"]:
                    case.update(_case_report)
                    return list_all
            
            module["TestCases"].append(_case_report)
            return list_all
        
        list_all.append({"Name": module_name, "TestCases": [_case_report]})
        return list_all