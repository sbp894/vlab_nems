#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:05:10 2021

@author: vatsun
"""

def main():
    from pathlib import Path
    import numpy as np
    import matplotlib
    from timeit import default_timer as timer
    matplotlib.use("Qt5Agg")
    
    starttime = timer()
    resultfilename = "D:/NEMS/results/results_summary.csv"
    resultfile = open(resultfilename,'a')
    np.savetxt(resultfile,["Response", "Model", "NParms", "Fit_Time", "r_fit",\
                           "se_fit", "r_test", "se_test", "mse_fit", "se_mse_fit",\
                               "mse_test","se_mse_test","ll_fit","ll_test",\
                                   "r_floor", "r_ceiling"], fmt = '%s', delimiter = ' ', newline = ', ')
    resultfile.write("\n")
    resultfile.close()
    
    lines = np.loadtxt("D:/NEMS/recordings/MS1001_1195/response_files_1195_5ms_test.txt", dtype=str, comments="#", delimiter=",", unpack=False)
    #modelids_test = ['fir.36x10-lvl.1-dexp.1']
    modelids_test = ['fir.36x10-lvl.1-dexp.1', 'wc.36x1.g-fir.1x10-lvl.1-dexp.1', 'wc.36x2.g-fir.2x10-lvl.1-dexp.1']
    #modelids_test = ['wc.36x1-fir.1x10-lvl.1-dexp.1', 'wc.36x2-fir.2x10-lvl.1-dexp.1']
        
    for respid in lines:
        for modelid in modelids_test:
            print(respid)
            outvals = vatsun_singlefile(str(respid), 'DRC_Stim1_Mouse_10Intens_sparse2_5ms.csv.gz', modelid)
            
            resultfile = open(resultfilename, 'a')
            np.savetxt(resultfile,[respid+" , "+modelid],fmt='%s', newline= ', ')
            np.savetxt(resultfile, outvals, fmt='%f', delimiter = ' ',newline = ', ')
            resultfile.write("\n")
            resultfile.close()

    endtime = timer()
    print(endtime-starttime)
    
    
            
def vatsun_singlefile(respid, stimid, modelid):
    import logging
    import pickle
    from pathlib import Path
    import gzip
    import numpy as np
    import pandas as pd
    import matplotlib
    import nems.analysis.api
    import nems.initializers
    import nems.recording as recording
    import nems.preprocessing as preproc
    import nems.uri
    from nems.fitters.api import scipy_minimize
    from nems.signal import RasterizedSignal
    import string

    log = logging.getLogger(__name__)
    
    # figure out data and results paths:
    signals_dir = Path(nems.NEMS_PATH) / 'recordings/MS1001_1195'
    modelspecs_dir = Path(nems.NEMS_PATH) / 'modelspecs'
    
    fs=200
    stimfile = signals_dir / stimid
    respfile = signals_dir / respid
    epochsfile = signals_dir / 'strf_epochs.csv'
    recname = str.split(respid,'_resp')[0]
    
    results_path = 'D:/NEMS/results/'
    figs_path = 'D:/NEMS/results/figures/'
    
    X=np.loadtxt(gzip.open(stimfile, mode='rb'), delimiter=",", skiprows=0)
    Y=np.loadtxt(gzip.open(respfile, mode='rb'), delimiter=",", skiprows=0)
    # get list of stimuli with start and stop times (in sec)
    epochs = pd.read_csv(epochsfile)
    
    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y, 'resp', recname, chans=[respid], epochs=epochs.loc[:])
    stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs.loc[:])
    
    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    rec = recording.Recording(signals)
    
    #generate est/val set_sets
    nfolds=10
    est = rec.jackknife_masks_by_time(njacks=nfolds, invert=False)
    val = rec.jackknife_masks_by_time(njacks=nfolds, invert=True)
    
    log.info('Initializing modelspec...')
    # record some meta data for display and saving
    meta = {'cellid': respid,
            'batch': 1,
            'modelname': modelid,
            'recording': est.name
            }
    modelspec = nems.initializers.from_keywords(modelid, meta=meta)
    if est.view_count>1:
        log.info('Tiling modelspec to match number of est views')
        modelspec.tile_jacks(nfolds)
    
    # RUN AN ANALYSIS
    
    # GOAL: Fit your model to your data, producing the improved modelspecs.
    #       Note that: nems.analysis.* will return a list of modelspecs, sorted
    #       in descending order of how they performed on the fitter's metric.
    
    log.info('Fitting model ...')
    
    for jack_idx, e in enumerate(est.views()):
        modelspec.jack_index = jack_idx
        log.info("----------------------------------------------------")
        log.info("Fitting: fold %d/%d", jack_idx + 1, modelspec.jack_count)
        
        if 'nonlinearity' in modelspec[-1]['fn']:
            # quick fit linear part first to avoid local minima
            modelspec = nems.initializers.prefit_LN(
                      e, modelspec, tolerance=1e-4, max_iter=500) #VATSUN- doesnt work if you change max_iter to 2000
    
            # uncomment to try TensorFlow pre-fitter:
            # modelspec = fit_tf_init(modelspec, e, epoch_name=None)['modelspec']
    
        # then fit full nonlinear model -- scipy fitter
        modelspec = nems.analysis.api.fit_basic(e, modelspec, fitter=scipy_minimize)
    
        # uncomment to try TensorFlow fitter:
        # modelspec = fit_tf(modelspec, e, epoch_name=None)['modelspec']
    
    # GENERATE SUMMARY STATISTICS
    log.info('Generating summary statistics ...')
    
    # generate predictions
    est, val = nems.analysis.api.generate_prediction(est, val, modelspec)
    
    # evaluate prediction accuracy
    modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)
    
    log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
            modelspec.meta['r_fit'][0][0],
            modelspec.meta['r_test'][0][0]))
    outvals = [modelspec.meta['n_parms'], modelspec.meta['fit_time'], \
               modelspec.meta['r_fit'][0][0], modelspec.meta['se_fit'][0][0], \
                   modelspec.meta['r_test'][0][0], modelspec.meta['se_test'][0][0], \
                   modelspec.meta['mse_fit'][0][0], modelspec.meta['se_mse_fit'][0][0], \
                       modelspec.meta['mse_test'][0][0], modelspec.meta['se_mse_test'][0][0], \
                           modelspec.meta['ll_fit'][0][0], modelspec.meta['ll_test'][0][0], \
                               modelspec.meta['r_floor'][0][0], modelspec.meta['r_ceiling'][0][0]]

    # SAVE YOUR RESULTS
    
    # uncomment to save model to disk
    logging.info('Saving Results...')
    savename = "file://" + results_path + recname + "_" + modelid + ".json"   
    # modelspec.save_modelspecs(modelspecs_dir, modelspec, 'test')
    nems.uri.save_resource(savename, json=modelspec.raw)
    #modelspec2=nems.modelspec.load_modelspec('file://tmp/test.json')
    
    # GENERATE PLOTS
    
    # GOAL: Plot the predictions made by your results vs the real response.
    #       Compare performance of results with other metrics.
    
    log.info('Generating summary plot ...')
    
    # uncomment to browse the validation data
    # from nems.gui.editors import EditorWindow
    # ex = EditorWindow(modelspec=modelspec, rec=val)
    
    # Generate a summary plot
    fig = modelspec.quickplot(rec=val)
    savefigname = figs_path + recname + "_" + modelid + ".svg"
    fig.savefig(savefigname, format = 'svg', transparent = True)
    matplotlib.pyplot.close()
    return outvals
  
if __name__ == "__main__":
    main()  # outputs: "inner function"
