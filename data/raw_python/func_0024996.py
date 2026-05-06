def download(url, path=None, headers=None, session=None, show_progress=True,
             resume=True, auto_retry=True, max_rst_retries=5,
             pass_through_opts=None, cainfo=None, user_agent=None, auth=None):
    """Main download function"""
    hm = Homura(url, path, headers, session, show_progress, resume,
                auto_retry, max_rst_retries, pass_through_opts, cainfo,
                user_agent, auth)
    hm.start()