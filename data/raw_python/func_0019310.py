def dirpath_int(self):
        """Absolute path of the directory of the internal data file.

        Normally, each sequence queries its current "internal" directory
        path from the |SequenceManager| object stored in module |pub|:

        >>> from hydpy import pub, repr_, TestIO
        >>> from hydpy.core.filetools import SequenceManager
        >>> pub.sequencemanager = SequenceManager()

        We overwrite |FileManager.basepath| and prepare a folder in teh
        `iotesting` directory to simplify the following examples:

        >>> basepath = SequenceManager.basepath
        >>> SequenceManager.basepath = 'test'
        >>> TestIO.clear()
        >>> import os
        >>> with TestIO():
        ...     os.makedirs('test/temp')

        Generally, |SequenceManager.tempdirpath| is queried:

        >>> from hydpy.core import sequencetools as st
        >>> seq = st.InputSequence(None)
        >>> with TestIO():
        ...     repr_(seq.dirpath_int)
        'test/temp'

        Alternatively, you can specify |IOSequence.dirpath_int| for each
        sequence object individually:

        >>> seq.dirpath_int = 'path'
        >>> os.path.split(seq.dirpath_int)
        ('', 'path')
        >>> del seq.dirpath_int
        >>> with TestIO():
        ...     os.path.split(seq.dirpath_int)
        ('test', 'temp')

        If neither an individual definition nor |SequenceManager| is
        available, the following error is raised:

        >>> del pub.sequencemanager
        >>> seq.dirpath_int
        Traceback (most recent call last):
        ...
        RuntimeError: For sequence `inputsequence` the directory of \
the internal data file cannot be determined.  Either set it manually \
or prepare `pub.sequencemanager` correctly.

        Remove the `basepath` mock:

        >>> SequenceManager.basepath = basepath
        """
        try:
            return hydpy.pub.sequencemanager.tempdirpath
        except RuntimeError:
            raise RuntimeError(
                f'For sequence {objecttools.devicephrase(self)} '
                f'the directory of the internal data file cannot '
                f'be determined.  Either set it manually or prepare '
                f'`pub.sequencemanager` correctly.')