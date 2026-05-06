def calc_signal_power(params):

    '''
    calculates power spectrum of sum signal for all channels

    '''

    for i, data_type in enumerate(['CSD','LFP','CSD_10_0', 'LFP_10_0']):
        if i % SIZE == RANK:

            # Load data
            if data_type in ['CSD','LFP']:
                fname=os.path.join(params.savefolder, data_type+'sum.h5')
            else:
                fname=os.path.join(params.populations_path, 'subsamples',
                                   str.split(data_type,'_')[0] + 'sum_' +
                                   str.split(data_type,'_')[1] + '_' +
                                   str.split(data_type,'_')[2] + '.h5')
            #open file
            f = h5py.File(fname)
            data = f['data'].value
            srate = f['srate'].value 
            tvec = np.arange(data.shape[1]) * 1000. / srate
        
            # slice
            slica = (tvec >= ana_params.transient)
            data = data[:,slica]
    
            # subtract mean
            dataT = data.T - data.mean(axis=1)
            data = dataT.T
            f.close()
    
            #extract PSD
            PSD=[]
            for i in np.arange(len(params.electrodeParams['z'])):
                if ana_params.mlab:
                    Pxx, freqs=plt.mlab.psd(data[i], NFFT=ana_params.NFFT,
                                        Fs=srate, noverlap=ana_params.noverlap,
                                        window=ana_params.window)
                else:
                    [freqs, Pxx] = hlp.powerspec([data[i]], tbin= 1.,
                                             Df=ana_params.Df, pointProcess=False)
                    mask = np.where(freqs >= 0.)
                    freqs = freqs[mask]
                    Pxx = Pxx.flatten()
                    Pxx = Pxx[mask]
                    Pxx = Pxx/tvec[tvec >= ana_params.transient].size**2
                PSD +=[Pxx.flatten()]
                
            PSD=np.array(PSD)
    
            # Save data
            f = h5py.File(os.path.join(params.savefolder, ana_params.analysis_folder,
                                       data_type + ana_params.fname_psd),'w')
            f['freqs']=freqs
            f['psd']=PSD
            f['transient']=ana_params.transient
            f['mlab']=ana_params.mlab
            f['NFFT']=ana_params.NFFT
            f['noverlap']=ana_params.noverlap
            f['window']=str(ana_params.window)
            f['Df']=str(ana_params.Df)
            f.close()

    return