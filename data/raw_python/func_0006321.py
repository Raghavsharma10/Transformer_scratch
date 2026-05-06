def select_hits_from_cluster_info(input_file_hits, output_file_hits, cluster_size_condition, n_cluster_condition, chunk_size=4000000):
    ''' Takes a hit table and stores only selected hits into a new table. The selection is done on an event base and events are selected if they have a certain number of cluster or cluster size.
    To increase the analysis speed a event index for the input hit file is created first. Since a cluster hit table can be created to this way of hit selection is
    not needed anymore.

     Parameters
    ----------
    input_file_hits: str
        the input file name with hits
    output_file_hits: str
        the output file name for the hits
    cluster_size_condition: str
        the cluster size condition to select events (e.g.: 'cluster_size_condition <= 2')
    n_cluster_condition: str
        the number of cluster in a event ((e.g.: 'n_cluster_condition == 1')
    '''
    logging.info('Write hits of events from ' + str(input_file_hits) + ' with ' + cluster_size_condition + ' and ' + n_cluster_condition + ' into ' + str(output_file_hits))
    with tb.open_file(input_file_hits, mode="r+") as in_hit_file_h5:
        analysis_utils.index_event_number(in_hit_file_h5.root.Hits)
        analysis_utils.index_event_number(in_hit_file_h5.root.Cluster)
        with tb.open_file(output_file_hits, mode="w") as out_hit_file_h5:
            hit_table_out = out_hit_file_h5.create_table(out_hit_file_h5.root, name='Hits', description=data_struct.HitInfoTable, title='hit_data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            cluster_table = in_hit_file_h5.root.Cluster
            last_word_number = 0
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=cluster_table.shape[0], term_width=80)
            progress_bar.start()
            for data, index in analysis_utils.data_aligned_at_events(cluster_table, chunk_size=chunk_size):
                selected_events_1 = analysis_utils.get_events_with_cluster_size(event_number=data['event_number'], cluster_size=data['size'], condition=cluster_size_condition)  # select the events with clusters of a certain size
                selected_events_2 = analysis_utils.get_events_with_n_cluster(event_number=data['event_number'], condition=n_cluster_condition)  # select the events with a certain cluster number
                selected_events = analysis_utils.get_events_in_both_arrays(selected_events_1, selected_events_2)  # select events with both conditions above
                logging.debug('Selected ' + str(len(selected_events)) + ' events with ' + n_cluster_condition + ' and ' + cluster_size_condition)
                last_word_number = analysis_utils.write_hits_in_events(hit_table_in=in_hit_file_h5.root.Hits, hit_table_out=hit_table_out, events=selected_events, start_hit_word=last_word_number)  # write the hits of the selected events into a new table
                progress_bar.update(index)
            progress_bar.finish()
            in_hit_file_h5.root.meta_data.copy(out_hit_file_h5.root)