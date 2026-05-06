def packing_job_ext_info(job_lsit_DO):
    """
    Packing additional information of the job into the job_list_DO(JobListDO)
    """
    ext_info = sqllite_agent.execute(ScrapydJobExtInfoSQLSet.SELECT_BY_ID, (job_lsit_DO.job_id,))
    if ext_info is None or len(ext_info) <= 0: return
    ext_info = ext_info[0]
    job_lsit_DO.args = ext_info[1]
    job_lsit_DO.priority = ext_info[2]
    job_lsit_DO.creation_time = ext_info[3]
    job_lsit_DO.logs_name = str_to_list(ext_info[4], ',')
    job_lsit_DO.logs_url = str_to_list(ext_info[5], ',')