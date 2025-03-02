import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata
import joblib

def MC(output_mat, rank_mat):
    output_mat = MinMaxScaler().fit_transform(rank_mat)
    
    m = output_mat.shape[1] #len(base_detectors) # number of candidate detectors
    
    similar_mat = np.full((m, m), 1).astype(float) # initialize the similarity matrix, B

    for k in range(output_mat.shape[1]): # for each detector
        temp_list = list(range(m))
        temp_list.remove(k)

        for j in temp_list:
            corr = kendalltau(output_mat[:, k], output_mat[:, j])[0]
            similar_mat[k, j] = corr
    
    B = (similar_mat+similar_mat.T)/2
    # fix nan problem
    B = np.nan_to_num(B)
    

    # Similarity is the IPM of the detectors
    similarity = (np.sum(B, axis=1)-1)/(m-1)    # Calculate model centrality
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')  

    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC1.kendall.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.kendall.csv'), similarity, delimiter=',') 
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'MC.csv'), similarity, delimiter=',')    
    # # y=pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR1.csv')).to_numpy()
    # # w = pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR.csv')).to_numpy()
    # # p = np.concatenate([w,y], axis=1)
    # # spearmanr(p)
    return similarity


def SELECT(score_mat, rank_mat):
    inv_rank_mat = 1 / rank_mat
    
    _, n_models = score_mat.shape[0], score_mat.shape[1]
    
    # build target vector 
    target = np.mean(inv_rank_mat, axis=1)
    
    kendall_vec = np.full([n_models,], -99).astype(float)
    kendall_tracker = []
    
    model_ind = list(range(n_models))
    selected_ind = []
    last_kendall = 0
    
    # build the first target
    for i in model_ind:
        kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
    
    most_sim_model = np.argmax(kendall_vec)
    kendall_tracker.append(np.max(kendall_vec))
    
    # option 3: average of all
    last_kendall = np.mean(kendall_tracker)
    
    selected_ind.append(most_sim_model)
    model_ind.remove(most_sim_model)
    
    
    while len(model_ind) != 0:
    
        target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
        kendall_vec = np.full([n_models,], -99).astype(float)
        
        for i in model_ind:
            kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
            
        most_sim_model = np.argmax(kendall_vec)
        max_kendall = np.max(kendall_vec)
        
        if max_kendall >= last_kendall:
            selected_ind.append(most_sim_model)
            model_ind.remove(most_sim_model)
            kendall_tracker.append(max_kendall)
            
            # option 3: average of all
            last_kendall = np.mean(kendall_tracker)

        else:
            break
    
    final_target = np.mean(inv_rank_mat[:, selected_ind], axis=1)

    similarity = []
    for k in range(n_models):
        similarity.append(kendalltau(final_target, inv_rank_mat[:, k])[0])
    
    return np.array(similarity), np.array(final_target*-1)
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Model.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Target.csv'), final_target*-1, delimiter=',')


def HITS(score_mat, rank_mat):
    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat
    
    n_samples, n_models = score_mat.shape[0], score_mat.shape[1]

    hub_vec = np.full([n_models, 1],  1/n_models)
    auth_vec = np.zeros([n_samples, 1])
    
    hub_vec_list = []
    auth_vec_list = []
    
    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)
    
    for i in range(500):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)
        
        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)
        
        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]
        
        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            print('break at', i)
            break
        
        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)
        

    return np.array([y[0] for y in hub_vec]), np.array([x[0] for x in auth_vec])  
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Model.csv'), hub_vec, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Target.csv'), auth_vec, delimiter=',')


def IPM(outlier_score):
    MC_s = []

    SELECT_s = []
    SELECT_t = []

    HITS_s = []
    HITS_t = []

    for scores in outlier_score:
        scores_mat = scores.to_numpy().astype('float64')
        scores_mat = np.nan_to_num(scores_mat)
        rank_mat = rankdata(scores_mat, axis=0)

        MC_s.append(MC(scores_mat, rank_mat))
        
        select_s, select_t = SELECT(scores_mat, rank_mat)
        SELECT_s.append(select_s)
        SELECT_t.append(select_t)

        hits_s, hits_t = HITS(scores_mat, rank_mat)
        HITS_s.append(hits_s)
        HITS_t.append(hits_t)
    
    return MC_s, SELECT_s, SELECT_t, HITS_s, HITS_t
