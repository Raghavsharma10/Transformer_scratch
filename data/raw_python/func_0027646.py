def get_all_results_from_jobs(user, j_id):
    """Get all results from job.
    """

    job = v1_utils.verify_existence_and_get(j_id, _TABLE)

    if not user.is_in_team(job['team_id']) and not user.is_read_only_user():
        raise dci_exc.Unauthorized()

    # get testscases from tests_results
    query = sql.select([models.TESTS_RESULTS]). \
        where(models.TESTS_RESULTS.c.job_id == job['id'])
    all_tests_results = flask.g.db_conn.execute(query).fetchall()

    results = []
    for test_result in all_tests_results:
        test_result = dict(test_result)
        results.append({'filename': test_result['name'],
                        'name': test_result['name'],
                        'total': test_result['total'],
                        'failures': test_result['failures'],
                        'errors': test_result['errors'],
                        'skips': test_result['skips'],
                        'time': test_result['time'],
                        'regressions': test_result['regressions'],
                        'successfixes': test_result['successfixes'],
                        'success': test_result['success'],
                        'file_id': test_result['file_id']})

    return flask.jsonify({'results': results,
                          '_meta': {'count': len(results)}})