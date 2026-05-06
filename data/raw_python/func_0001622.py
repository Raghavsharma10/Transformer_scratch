def get_summary(list_all=[], **kwargs):
        ''' summarize the report data
            @param list_all: a list which save the report data
            @param kwargs: such as
                show_all:    True/False   report show all status cases
                proj_name:   project name 
                home_page:   home page url
        
        '''
        all_summary = []
               
        for module in list_all:
            summary = {
                        "module_name" : module['Name'],
                        "show_all" : kwargs.get("show_all",True),
                        "project_name" : kwargs.get("proj_name","TestProject"),
                        "home_page" : kwargs.get("home_page",__about__.HOME_PAGE),
                        "start_time" : "",
                        "end_time" : "",
                        "duration_seconds" : "",
                        "total_case_num" : len(module["TestCases"]),
                        "pass_cases_num" : 0,
                        "fail_cases_num" : 0,
                        "details" : []
                    }
                         
            for case in module["TestCases"]:
                case_detail = {}
                case_detail["linkurl"] =  "./caselogs/%s_%s.log" %(case["case_name"],case["exec_date"])
                
                if case["status"].lower() == "pass":
                    summary["pass_cases_num"] += 1
                    case_detail["c_style"] = "tr_pass"
                else:
                    summary["fail_cases_num"] += 1
                    case_detail["c_style"] = "tr_fail"
                
                case_detail.update(case)
            
                summary["details"].append(case_detail)                       
             
            try:
                st = module["TestCases"][0].get("start_at")
                et = module["TestCases"][-1].get("end_at")
                
                summary["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(st))    
                summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(et))        
                summary["duration_seconds"] = float("%.2f" %(et - st))
            except Exception as _:
                logger.log_warning("Will set 'start_at' and 'end_at' to 'None'")
                (summary["start_time"], summary["end_time"], summary["duration_seconds"]) = (None,None,None)
                    
            if summary["fail_cases_num"] > 0:
                summary["dict_report"] = {"result":0,"message":"failure","pass":summary["pass_cases_num"],"fail":summary["fail_cases_num"]}
            else:
                summary["dict_report"] = {"result":1,"message":"success","pass":summary["pass_cases_num"],"fail":summary["fail_cases_num"]}
             
            all_summary.append(summary)
            
        return all_summary