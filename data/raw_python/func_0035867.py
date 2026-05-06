def download_matches(match_downloaded_callback, on_exit_callback, conf, synchronize_callback= True):
    """
    :param match_downloaded_callback:       function       when a match is downloaded function is called with the match
                                                            and the tier (league) of the lowest player in the match
                                                            as parameters

    :param on_exit_callback:                function        when this function is terminating on_exit_callback is called
                                                            with the remaining players to download, the downloaded
                                                            players, the id of the remaining matches to download and
                                                            the id of the downloaded matches

    :param conf:                            dict           a dictionary containing all the configuration parameters

    :param synchronize_callback:            bool            Synchronize the calls to match_downloaded_callback
                                                            If set to True the calls are wrapped by a lock, so that only
                                                            one at a time is executing

    :return:                                None
    """

    logger = logging.getLogger(__name__)
    if conf['logging_level'] != logging.NOTSET:
        logger.setLevel(conf['logging_level'])
    else:
        # possibly set the level to warning
        pass

    def checkpoint(players_to_analyze, analyzed_players, matches_to_download, downloaded_matches):
        logger.info("Reached the checkpoint."
                    .format(datetime.datetime.now().strftime("%m-%d %H:%M:%S"), len(downloaded_matches)))
        if on_exit_callback:
            on_exit_callback(players_to_analyze, analyzed_players, matches_to_download, downloaded_matches)

    players_to_analyze = set(conf['seed_players_id'])
    downloaded_matches = set(conf['downloaded_matches'])
    logger.info("{} previously downloaded matches".format(len(downloaded_matches)))
    matches_to_download = set(conf['matches_to_download'])
    logger.info("{} matches to download".format(len(matches_to_download)))

    analyzed_players = set()
    pta_lock = threading.Lock()
    players_available_condition = threading.Condition(pta_lock)
    mtd_lock = threading.Lock()
    matches_Available_condition = threading.Condition(mtd_lock)
    user_function_lock = threading.Lock() if synchronize_callback else NoOpContextManager()
    logger_lock = threading.Lock()
    player_downloader_threads = []
    match_downloader_threads = []

    try:

        def create_thread():
            if len(player_downloader_threads) < max_players_download_threads:
                player_downloader = PlayerDownloader(conf, players_to_analyze, analyzed_players, pta_lock, players_available_condition,
                                         matches_to_download , mtd_lock, matches_Available_condition,
                                         logger, logger_lock)
                player_downloader.start()
                player_downloader_threads.append(player_downloader)
                with logger_lock:
                    logger.info("Adding a player download thread. Threads: " + str(len(player_downloader_threads)))
            else:
                with logger_lock:
                    logger.debug("Tried adding a player download thread, but there are already the maximum number:"
                                " " + str(max_players_download_threads))

        def shutdown_thread():
            if len(player_downloader_threads) > 1:
                player_downloader_threads.pop().shutdown()
                with logger_lock:
                    logger.info("Removing a player downloader thread. Threads: " + str(len(player_downloader_threads)))
            else:
                with logger_lock:
                    logger.debug("Tried removing a player download thread, but there is only one left")


        logger.info("Starting fetching..")
        # Start one player downloader thread
        create_thread()

        for _ in range(matches_download_threads):
            match_downloader = MatchDownloader(conf, players_to_analyze, pta_lock, players_available_condition,
                                               matches_to_download, downloaded_matches, mtd_lock, matches_Available_condition,
                                               match_downloaded_callback, user_function_lock,
                                               logger, logger_lock)
            match_downloader.start()
            match_downloader_threads.append(match_downloader)

        auto_tuner = ThreadAutoTuner(create_thread, shutdown_thread)

        for i, _ in enumerate(do_every(1)):
            # Pool the exit flag every second
            if conf.get('exit', False):
                break

            if i % 5 == 0:
                with mtd_lock:
                    matches_in_queue = len(matches_to_download)

                # The lock happens in the property. Since it is not re-entrant, do not lock now
                total_players = sum(th.total_downloads for th in player_downloader_threads)

                auto_tuner.update_thread_number(total_players, matches_in_queue)

            # Execute every LOGGING_INTERVAL seconds
            if i % logging_interval == 0:
                with mtd_lock:
                    matches_in_queue = len(matches_to_download)
                total_matches = sum(th.total_downloads for th in match_downloader_threads)
                with pta_lock:
                    players_in_queue = len(players_to_analyze)
                total_players = sum(th.total_downloads for th in player_downloader_threads)
                with logger_lock:
                    logger.info("Players in queue: {}. Downloaded players: {}. Matches in queue: {}. Downloaded matches: {}"
                                    .format(players_in_queue, total_players, matches_in_queue, total_matches))

        # Notify all the waiting threads so they can exit
        with pta_lock:
            players_available_condition.notify_all()
        with mtd_lock:
            matches_Available_condition.notify_all()
        logger.info("Terminating fetching")

    finally:
        conf['exit'] = True
        # Joining threads before saving the state
        for thread in player_downloader_threads + match_downloader_threads:
            thread.join()
        # Always call the checkpoint, so that we can resume the download in case of exceptions.
        logger.info("Calling checkpoint callback")
        checkpoint(players_to_analyze, analyzed_players, matches_to_download, downloaded_matches)