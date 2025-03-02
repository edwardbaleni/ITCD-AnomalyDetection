import utils.metaTrain as mt
import joblib
from joblib import Parallel, delayed

def run(model_name):
    perf, data, HPs, study = mt.loadPerformance(model_name)

    MC, SELECT_s, SELECT_t, HITS_s, HITS_t, output = mt.getIPM(perf, data, model_name)


    meta_dataframe, y_meta = mt.getMetaFeatures(data, output, perf, HPs, study, model_name, MC, SELECT_s, HITS_s)

    joblib.dump(meta_dataframe, f'results/meta/{model_name}/meta_dataframe.pkl')
    joblib.dump(y_meta, f'results/meta/{model_name}/y_meta.pkl')

    mt.trainPPE(data, meta_dataframe, y_meta, model_name)
    mt.train_full(meta_dataframe, y_meta, model_name)
    # Save
        # meta-feature extractor
        # IPM extractor
        # PPE
        # MSF


# XXX: Train the meta-surrogate funcion h(I, g)
# Done in testing file

if __name__ == "__main__":
    files = ["LOF", "ABOD", "IF"]#,PCA"]

    Parallel(n_jobs=3)(
        delayed(run)(model) for model in files
    )