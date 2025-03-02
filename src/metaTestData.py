import joblib
import pandas as pd
from os import listdir
from sklearn.preprocessing import RobustScaler
from utils.IPM import IPM
from utils.gen_meta_features import generate_meta_features
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from joblib import Parallel, delayed

def getMetaTestData(model_name):
    def loadData(model_name):
        data = joblib.load("results/testing/data70_101.pkl")
        # supp = joblib.load("results/testing/supp70_101.pkl")

        # Get hyperparameters

            
        path = "results/hyperparameter/"
        home = listdir(path)


        # either ABOD, LOF, PCA, or IF
        files = [f'{path}{file}' for file in home if model_name in file]

        study = [joblib.load(file) for file in files]

        study = study[0]

        perf = study.trials_dataframe()
        HPs = []
        HP = None

        if model_name == 'LOF':    
            HP = perf.loc[:, ["params_metric", "params_n_neighbors"]] 
            HPs = ['params_n_neighbors', 'params_metric']
        elif model_name == 'ABOD':
            HP = perf.loc[:, 'params_n_neighbors']
            HPs = ['params_n_neighbors']
        elif model_name == 'PCA':
            HP = perf.loc[:,'params_n_selected_components']
            HPs = ['params_n_selected_components']
        elif model_name == 'IF':
            HP = perf.loc[:, ['params_max_features', 'params_n_estimators']]
            HPs = ['params_max_features', 'params_n_estimators']


        for i in range(len(data)):
            data[i].loc[:,"confidence":] = RobustScaler().fit_transform(data[i].loc[:, "confidence":])
        
        return data, HP, HPs

    # Get IPM

    def getIPM(data, output, model_name):

        outlier_score = []

        from pyod.models.lof import LOF
        from pyod.models.abod import ABOD
        from pyod.models.pca import PCA
        from pyod.models.iforest import IForest
        i = 70
        for x in data:
            scores = pd.DataFrame()
            for j in range(output.shape[0]):
                # get the hyperparameters at the jth index
                if model_name == 'LOF':
                    neighbour = (output['params_n_neighbors'].iloc[j])
                    metric = (output['params_metric'].iloc[j])
                    clf = LOF(n_neighbors=neighbour, metric=metric)
                elif model_name == 'ABOD':
                    neighbour = (output.iloc[j])
                    clf = ABOD(n_neighbors=neighbour)
                elif model_name == 'PCA':
                    n_components = (output.iloc[j])
                    clf = PCA(n_components=n_components)
                elif model_name == 'IF':
                    max_features = (output['params_max_features'].iloc[j])
                    n_estimators = (output['params_n_estimators'].iloc[j])
                    clf = IForest(max_features=max_features, n_estimators=n_estimators)
                
                clf.fit(x.loc[:, "confidence":])
                y_test_scores = clf.decision_scores_
                
                if model_name == 'LOF':
                    scores[f"(LOF, {(neighbour, metric)})"] = y_test_scores
                elif model_name == 'ABOD':
                    scores[f"(ABOD, {(neighbour)})"] = y_test_scores
                elif model_name == 'PCA':
                    scores[f"(PCA, {(n_components)})"] = y_test_scores
                elif model_name == 'IF':
                    scores[f"(IF, {(max_features, n_estimators)})"] = y_test_scores

            
            print(f'Orchard {i+1} done')
            i += 1
            outlier_score.append(scores)

        MC, SELECT_s, SELECT_t, HITS_s, HITS_t = IPM(outlier_score)
        return MC, SELECT_s, SELECT_t, HITS_s, HITS_t

    # Get the meta-features
    # extract meta-features per task g(X)

    def getMetaFeatures(data, output, HPs, model_name, MC, SELECT_s, HITS_s):
        # feature_list = feats+HPs+['MC', 'SELECT', 'HITS']
        meta_dataframe = pd.DataFrame()#columns=feature_list)

        # Binarize the 'params_metric' column
        if model_name == 'LOF':
            lb = LabelBinarizer()
            binarize = lb.fit_transform(output['params_metric'])

            metric_df = pd.DataFrame(binarize, 
                                    columns = lb.classes_) 
            
        # 0 isnt working very well right now
        # TODO: Change the 38 to len(data)
        for i in range(len(data)):
            meta_vals, meta_feats = generate_meta_features(data[i].loc[:, "confidence":])
            meta_vals = np.nan_to_num(meta_vals, copy=False)

            meta_vals = np.tile(meta_vals.T, MC[i].shape[0])
            meta_vals = meta_vals.reshape(MC[i].shape[0], len(meta_feats))

            # HP = [int(re.findall(r'\d+', j)[0]) for j in outlier_score[i].columns]
            if model_name == 'LOF':
                meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                                    # pd.DataFrame({'params_n_neighbors': HP}),
                                                output.loc[:, ['params_n_neighbors']],
                                                metric_df,
                                                pd.DataFrame({'MC'    : MC[i]}), 
                                                pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                                pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
            elif model_name == 'ABOD':
                meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                                pd.DataFrame({'params_n_neighbors': output}),
                                                pd.DataFrame({'MC'    : MC[i]}), 
                                                pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                                pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
            elif model_name == 'PCA':
                meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                                pd.DataFrame({'params_n_selected_components':output}),
                                                pd.DataFrame({'MC'    : MC[i]}), 
                                                pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                                pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
            elif model_name == 'IF':
                meta_ = pd.concat([pd.DataFrame(data = meta_vals, columns=meta_feats), 
                                                output.loc[:, ['params_max_features', 'params_n_estimators']],
                                                pd.DataFrame({'MC'    : MC[i]}), 
                                                pd.DataFrame({'SELECT':SELECT_s[i]}), 
                                                pd.DataFrame({'HITS'  : HITS_s[i]})], axis=1)
            meta_['Dataset'] = i
            
            meta_dataframe = pd.concat([meta_dataframe, meta_])

        y_meta = pd.DataFrame()
        cols = ['Dataset'] + [col for col in meta_.columns if col != 'Dataset']
        meta_dataframe = meta_dataframe[cols]



            

        meta_dataframe.reset_index(drop=True, inplace=True)
        if model_name == 'LOF' or model_name == 'ABOD':
            meta_dataframe['params_n_neighbors'] = meta_dataframe['params_n_neighbors'].astype(int)
        elif model_name == 'PCA':
            meta_dataframe['params_n_selected_components'] = meta_dataframe['params_n_selected_components'].astype(int)
        elif model_name == 'IF':
            meta_dataframe['params_max_features'] = meta_dataframe['params_max_features'].astype(int)
            meta_dataframe['params_n_estimators'] = meta_dataframe['params_n_estimators'].astype(int)

        # meta_dataframe.drop(columns=['Dataset'], inplace=True)

        return meta_dataframe

    data, output, HPs = loadData(model_name)
    MC, SELECT_s, _, HITS_s, _ = getIPM(data, output, model_name)
    print(output)
    meta_dataframe = getMetaFeatures(data, output, HPs, model_name, MC, SELECT_s, HITS_s)

    joblib.dump(meta_dataframe, f'results/meta/Test/{model_name}_meta_dataframe.pkl')

    # return meta_dataframe

# cannot train PPE because that is only for training

if __name__ == "__main__":
    # model_name = "PCA"
    # joblib.dump(getMetaTestData(model_name), f'results/meta/Test/{model_name}/meta_dataframe.pkl')

    files = ["LOF", "ABOD", "IF", "PCA"]

    Parallel(n_jobs=4)(
        delayed(getMetaTestData)(model) for model in files
    )