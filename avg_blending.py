from __future__ import division
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

def xgbPredict(trainFeature,trainLabel,testFeature,rounds,params):
    #params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
    dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
    dtest = xgb.DMatrix(testFeature, label = np.zeros(testFeature.shape[0]))

    watchlist  = [(dtrain,'train')]
    num_round = rounds
    
    model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 500)
    predict = model.predict(dtest)
    return model, predict

def getBlending(train,test,xgb_params,lgb_params,xgb_rounds,lgb_rounds):
	train = pd.read_csv(FEATURE_PATH+train)
	test = pd.read_csv(FEATURE_PATH+test)
	trainFeature = train.drop(['EID','TARGET'],axis=1)
	testFeature = test.drop('EID',axis=1)
	trainLabel = train['TARGET'].values
	lgb_train = lgb.Dataset(trainFeature, trainLabel)
	xgb_res = []
	lgb_res = []
	for SEED in range(16,24):
	    xgb_params['seed'] = SEED
	    lgb_params['seed'] = SEED
	    #X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=0.1, random_state=SEED)
	    print 'training xgb..',SEED
	    model,pred1 = xgbPredict(trainFeature,trainLabel,testFeature,xgb_rounds,xgb_params)
	    xgb_res.append(pred1)
	    print 'training lgb..',SEED
	    LGBmodel = lgb.train(lgb_params,lgb_train,num_boost_round=lgb_rounds,verbose_eval=0)
	    pred2 = LGBmodel.predict(testFeature)
	    lgb_res.append(pred2)

	xgb_pred = np.array(xgb_res).mean(axis=0)
	lgb_pred = np.array(lgb_res).mean(axis=0)
	res = 0.6*xgb_pred + 0.4*lgb_pred
	return res

def storeResult(testIndex,preds,threshold,name,):
    result = pd.DataFrame({'EID':testIndex,'FORTARGET':0,'PROB':preds})
    mask = result['PROB'] >= threshold
    result.at[mask,'FORTARGET'] = 1
    result['PROB'] = result['PROB'].apply(lambda x:round(x,4))
    result.to_csv(RESULT_PATH+name+'.csv',index=0)
    return result

xgb_params1 = {
    'booster':'gbtree',
    #'objective':'rank:pairwise',
    'objective':'binary:logistic',
    'stratified':True,
    #'scale_pos_weights ':0,
    'max_depth':7,
    'min_child_weight':1,
    'gamma':1,
    'subsample':0.8,
    'colsample_bytree':0.75,
    'lambda':3,
    
    'eta':0.02,
    'seed':20,
    'silent':1,
    'eval_metric':'auc',
    'tree_method':'gpu_hist'
}

xgb_params2 = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'stratified':True,
    #'scale_pos_weights ':0,
    #'max_depth':6,
    'max_depth':8,
    'min_child_weight':1,
    'gamma':4,
    'subsample':0.7,#0.7
    'colsample_bytree':0.6,
    #'lambda':1,   
    'lambda':3, 
    'eta':0.02,
    'seed':20,
    'silent':1,
    'eval_metric':'auc',
    'tree_method':'gpu_hist'
}

xgb_params3 = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'stratified':True,
    'max_depth':8,
    'min_child_weight':0.5,
    'gamma':2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    
    #'lambda':0.001,   #550
#     'alpha':0.00001,
#     'lambda_bias':0.1,
    #'threads':512,
    'eta': 0.02,
    'seed':42, 
    'silent': 1,
    'tree_method':'gpu_hist'
}

lgb_params1 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'maxdepth':-1,
    'metric': 'auc',
    'num_leaves': 128,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 6,
    'verbosity': 2,
    'tree_learner':'feature',
    'min_sum_hessian_in_leaf':0.1,
    #'min_data_in_leaf':15
    #'max_bin':
}

lgb_params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'maxdepth':-1,
    'metric': 'auc',
    'num_leaves': 128,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 6,
    'verbosity': 2,
    'tree_learner':'feature',
    'min_sum_hessian_in_leaf':0.1,
    #'min_data_in_leaf':15
    #'max_bin':
}

lgb_params3 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'maxdepth':-1,
    'metric': 'auc',
    'num_leaves': 128,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 6,
    'verbosity': 2,
    'tree_learner':'feature',
    'min_sum_hessian_in_leaf':0.1,
    #'min_data_in_leaf':15
    #'max_bin':
}



if __name__ == '__main__':

	FEATURE_PATH = 'feature/'
	RESULT_PATH = 'result/'

	trainFeature_files = ['trainFeature_zx_6928_withEID.csv', 'train_v1.6.csv']
	testFeature_files = ['testFeature_zx_6928_withEID.csv', 'test_v1.6.csv']
	xgb_param = [xgb_params1, xgb_params2, xgb_params3]
	lgb_param = [lgb_params1, lgb_params2, lgb_params3]
	xgb_rounds = [1293, 1136, 794]
	lgb_rounds = [395, 474, 434]
	result = []
	for i in range(3):
		res = getBlending(trainFeature_files[i],testFeature_files[i],xgb_param[i],lgb_param[i],xgb_rounds[i],lgb_rounds[i])
		result.append(res)

	pred = result[0]*0.6 + result[1]*0.4
	testIndex = pd.read_csv(FEATURE_PATH+testFeature_files[0]).EID.values

	storeResult(testIndex,pred,0.16,'avg_blending')
