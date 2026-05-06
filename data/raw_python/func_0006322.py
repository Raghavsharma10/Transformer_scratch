def select_hits(input_file_hits, output_file_hits, condition=None, cluster_size_condition=None, n_cluster_condition=None, chunk_size=5000000):
    ''' Takes a hit table and stores only selected hits into a new table. The selection of hits is done with a numexp string. Only if
    this expression evaluates to true the hit is taken. One can also select hits from cluster conditions. This selection is done
    on an event basis, meaning events are selected where the cluster condition is true and then hits of these events are taken.

     Parameters
    ----------
    input_file_hits: str
        the input file name with hits
    output_file_hits: str
        the output file name for the hits
    condition: str
        Numexpr string to select hits (e.g.: '(relative_BCID == 6) & (column == row)')
        All hit infos can be used (column, row, ...)
    cluster_size_condition: int
        Hit of events with the given cluster size are selected.
    n_cluster_condition: int
        Hit of events with the given cluster number are selected.
    '''
    logging.info('Write hits with ' + condition + ' into ' + str(output_file_hits))
    if cluster_size_condition is None and n_cluster_condition is None:  # no cluster cuts are done
        with tb.open_file(input_file_hits, mode="r+") as in_hit_file_h5:
            analysis_utils.index_event_number(in_hit_file_h5.root.Hits)  # create event index for faster selection
            with tb.open_file(output_file_hits, mode="w") as out_hit_file_h5:
                hit_table_out = out_hit_file_h5.create_table(out_hit_file_h5.root, name='Hits', description=data_struct.HitInfoTable, title='hit_data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                analysis_utils.write_hits_in_event_range(hit_table_in=in_hit_file_h5.root.Hits, hit_table_out=hit_table_out, condition=condition)  # write the hits of the selected events into a new table
                in_hit_file_h5.root.meta_data.copy(out_hit_file_h5.root)  # copy meta_data note to new file
    else:
        with tb.open_file(input_file_hits, mode="r+") as in_hit_file_h5:  # open file with hit/cluster data with r+ to be able to create index
            analysis_utils.index_event_number(in_hit_file_h5.root.Hits)  # create event index for faster selection
            analysis_utils.index_event_number(in_hit_file_h5.root.Cluster)  # create event index for faster selection
            with tb.open_file(output_file_hits, mode="w") as out_hit_file_h5:
                hit_table_out = out_hit_file_h5.create_table(out_hit_file_h5.root, name='Hits', description=data_struct.HitInfoTable, title='hit_data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                cluster_table = in_hit_file_h5.root.Cluster
                last_word_number = 0
                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=cluster_table.shape[0], term_width=80)
                progress_bar.start()
                for data, index in analysis_utils.data_aligned_at_events(cluster_table, chunk_size=chunk_size):
                    if cluster_size_condition is not None:
                        selected_events = analysis_utils.get_events_with_cluster_size(event_number=data['event_number'], cluster_size=data['size'], condition='cluster_size == ' + str(cluster_size_condition))  # select the events with only 1 hit cluster
                        if n_cluster_condition is not None:
                            selected_events_2 = analysis_utils.get_events_with_n_cluster(event_number=data['event_number'], condition='n_cluster == ' + str(n_cluster_condition))  # select the events with only 1 cluster
                            selected_events = selected_events[analysis_utils.in1d_events(selected_events, selected_events_2)]  # select events with the first two conditions above
                    elif n_cluster_condition is not None:
                        selected_events = analysis_utils.get_events_with_n_cluster(event_number=data['event_number'], condition='n_cluster == ' + str(n_cluster_condition))
                    else:
                        raise RuntimeError('Cannot understand cluster selection criterion')
                    last_word_number = analysis_utils.write_hits_in_events(hit_table_in=in_hit_file_h5.root.Hits, hit_table_out=hit_table_out, events=selected_events, start_hit_word=last_word_number, condition=condition, chunk_size=chunk_size)  # write the hits of the selected events into a new table
                    progress_bar.update(index)
                progress_bar.finish()
                in_hit_file_h5.root.meta_data.copy(out_hit_file_h5.root)