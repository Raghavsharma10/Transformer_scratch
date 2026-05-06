def load_data_split(proc_data_dir):
    """Loads a split dataset

        Args:
            proc_data_dir: Directory with the split and processed data

        Returns:
            (Training Data, Validation Data, Test Data)
    """
    ds_train = Dataset.load(path.join(proc_data_dir, 'train.bin'))
    ds_val = Dataset.load(path.join(proc_data_dir, 'val.bin'))
    ds_test = Dataset.load(path.join(proc_data_dir, 'test.bin'))
    return ds_train, ds_val, ds_test