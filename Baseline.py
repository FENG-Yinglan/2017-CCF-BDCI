import numpy as np
import pandas as pd
import xgboost as xgb
import catboost
import lightgbm as lgb
from dateutil import parser
from datetime import datetime

# reading data
train = pd.read_csv('/data/BDCI2017/riskPrediction/data/train.csv')
test = pd.read_csv('/data/BDCI2017/riskPrediction/data/evaluation_public.csv')
entbase = pd.read_csv('/data/BDCI2017/riskPrediction/data/1entbase.csv')
alter = pd.read_csv('/data/BDCI2017/riskPrediction/data/2alter.csv')
branch = pd.read_csv('/data/BDCI2017/riskPrediction/data/3branch.csv')
invest = pd.read_csv('/data/BDCI2017/riskPrediction/data/4invest.csv')
right = pd.read_csv('/data/BDCI2017/riskPrediction/data/5right.csv')
project = pd.read_csv('/data/BDCI2017/riskPrediction/data/6project.csv')
lawsuit = pd.read_csv('/data/BDCI2017/riskPrediction/data/7lawsuit.csv')
breakfaith = pd.read_csv('/data/BDCI2017/riskPrediction/data/8breakfaith.csv')
recruit = pd.read_csv('/data/BDCI2017/riskPrediction/data/9recruit.csv')

#entbase features
train_entbase = pd.merge(train, entbase, on='EID', how='left')
train_entbase['RGYEAR'] = 2017 - train_entbase.RGYEAR
train_entbase.MPNUM.fillna(0, inplace=True)
train_entbase.INUM.fillna(0, inplace=True)
train_entbase.FINZB.fillna(0, inplace=True)
train_entbase.FSTINUM.fillna(0, inplace=True)
train_entbase.TZINUM.fillna(0, inplace=True)
del train_entbase['TARGET']

test_entbase = pd.merge(test, entbase, on='EID', how='left')
test_entbase['RGYEAR'] = 2017 - test_entbase.RGYEAR
test_entbase.MPNUM.fillna(0, inplace=True)
test_entbase.INUM.fillna(0, inplace=True)
test_entbase.FINZB.fillna(0, inplace=True)
test_entbase.FSTINUM.fillna(0, inplace=True)
test_entbase.TZINUM.fillna(0, inplace=True)


#alter features
#alter features
from sklearn import preprocessing
alter = pd.read_csv('/data/BDCI2017/riskPrediction/data/2alter.csv')
le = preprocessing.LabelEncoder()
alter.ALTERNO = le.fit_transform(alter.ALTERNO.values)
alter = alter.drop_duplicates()
alter['ALTYEARDEL'] = 2017 - pd.to_datetime(alter.ALTDATE).dt.year
alter['ALTMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(alter.ALTDATE)).dt.days // 30
altbe = []
altaf = []
for i in alter.ALTBE.fillna(-1):
    if i == -1:
        altbe.append(-1)
    elif i[-6:] == '万元':
        if i[:-6] == 'null':
            altbe.append(-1)
        else:
            altbe.append(float(i[:-6]))
    else:
        altbe.append(float(i))
        
for i in alter.ALTAF.fillna(-1):
    if i == -1:
        altaf.append(-1)
    elif i[-6:] == '万元':
        altaf.append(float(i[:-6]))
    else:
        altaf.append(float(i))

alter['ALTBE'] = altbe
alter['ALTAF'] = altaf
alter.ALTBE.replace(-1, np.nan, inplace=True)
alter.ALTAF.replace(-1, np.nan, inplace=True)
alter['ALTBADEL'] = alter.ALTAF - alter.ALTBE
alter_feat = alter.groupby('EID', as_index=0)['ALTERNO'].count()
alter_feat.columns = ['EID', 'ALT_NUM']
alter_feat['ALTBE_MIN'] = alter.groupby('EID')['ALTBE'].min()
alter_feat['ALTBE_MAX'] = alter.groupby('EID')['ALTBE'].max()
alter_feat['ALTBE_MEAN'] = alter.groupby('EID')['ALTBE'].mean()
alter_feat['ALTAF_MIN'] = alter.groupby('EID')['ALTAF'].min()
alter_feat['ALTAF_MAX'] = alter.groupby('EID')['ALTAF'].max()
alter_feat['ALTAF_MEAN'] = alter.groupby('EID')['ALTAF'].mean()
alter_feat['ALTYEARDEL_MIN'] = alter.groupby('EID')['ALTYEARDEL'].min()
alter_feat['ALTYEARDEL_MAX'] = alter.groupby('EID')['ALTYEARDEL'].max()
alter_feat['ALTYEARDEL_MEAN'] = alter.groupby('EID')['ALTYEARDEL'].mean()
alter_feat['ALTMONDEL_MIN'] = alter.groupby('EID')['ALTMONDEL'].min()
alter_feat['ALTMONDEL_MAX'] = alter.groupby('EID')['ALTMONDEL'].max()
alter_feat['ALTMONDEL_MEAN'] = alter.groupby('EID')['ALTMONDEL'].mean()
alter_feat['ALTBADEL_MIN'] = alter.groupby('EID')['ALTBADEL'].min()
alter_feat['ALTBADEL_MAX'] = alter.groupby('EID')['ALTBADEL'].max()
alter_feat['ALTBADEL_MEAN'] = alter.groupby('EID')['ALTBADEL'].mean()
alter_feat.fillna(-1, inplace=True)
train_alter = pd.merge(train, alter_feat, on='EID', how='left').fillna(0)
test_alter = pd.merge(test, alter_feat, on='EID', how='left').fillna(0)
train_alter.replace(np.nan, -1, inplace=True)
test_alter.replace(np.nan, -1, inplace=True)
del train_alter['TARGET']

#branch features
branch = branch.drop_duplicates()
branch['B_YEARGAP'] = branch.B_ENDYEAR - branch.B_REYEAR
branch_feat = branch.groupby('EID', as_index=0)['TYPECODE'].count()
branch_feat.rename(columns={'TYPECODE':'branch_num'}, inplace=True)
branch_feat['IFHOME_NUM'] = branch.groupby('EID')['IFHOME'].sum().values
branch_feat['B_REYEARMIN'] = branch.groupby('EID')['B_REYEAR'].min().values
branch_feat['B_REYEARMAX'] = branch.groupby('EID')['B_REYEAR'].max().values
branch_feat['B_YEARGAPMIN'] = branch.groupby('EID')['B_YEARGAP'].min().values
branch_feat['B_YEARGAPMAX'] = branch.groupby('EID')['B_YEARGAP'].max().values
branch_feat['B_CLOSE_NUM'] = branch.groupby('EID')['B_ENDYEAR'].count().values
branch_feat['B_CLOSE_RATE'] = branch_feat.B_CLOSE_NUM / branch_feat.branch_num

train_branch = pd.merge(train, branch_feat, on='EID', how='left')
test_branch = pd.merge(test, branch_feat, on='EID', how='left')
del train_branch['TARGET']


#invest features
invest = invest.drop_duplicates()
train_invest = train.drop('TARGET', axis=1)
tmp = invest.groupby('EID', as_index=0)['BTEID'].count().rename(columns={'BTEID':'INVEST_NUM'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['EID'].count().rename(columns={'BTEID':'EID', 'EID':'BT_NUM'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTENDYEAR'].count().rename(columns={'BTENDYEAR':'INVESTLIVE_NUM'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
train_invest['INVESTLIVE_RATE'] = train_invest.INVESTLIVE_NUM / train_invest.INVEST_NUM
train_invest.INVESTLIVE_RATE.fillna(0, inplace=True)
tmp = invest.groupby('EID', as_index=0)['BTBL'].min().rename(columns={'BTBL':'BTBL_MIN'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].max().rename(columns={'BTBL':'BTBL_MAX'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].mean().rename(columns={'BTBL':'BTBL_MEAN'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].sum().rename(columns={'BTBL':'BTBL_SUM'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].min().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MIN'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].max().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MAX'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].mean().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MEAN'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].sum().rename(columns={'BTEID':'EID','BTBL':'BTBLED_SUM'})
train_invest = pd.merge(train_invest, tmp, on='EID', how='left').fillna(0)


test_invest = test.copy()
tmp = invest.groupby('EID', as_index=0)['BTEID'].count().rename(columns={'BTEID':'INVEST_NUM'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['EID'].count().rename(columns={'BTEID':'EID', 'EID':'BT_NUM'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTENDYEAR'].count().rename(columns={'BTENDYEAR':'INVESTLIVE_NUM'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
test_invest['INVESTLIVE_RATE'] = test_invest.INVESTLIVE_NUM / test_invest.INVEST_NUM
test_invest.INVESTLIVE_RATE.fillna(0, inplace=True)
tmp = invest.groupby('EID', as_index=0)['BTBL'].min().rename(columns={'BTBL':'BTBL_MIN'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].max().rename(columns={'BTBL':'BTBL_MAX'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].mean().rename(columns={'BTBL':'BTBL_MEAN'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('EID', as_index=0)['BTBL'].sum().rename(columns={'BTBL':'BTBL_SUM'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].min().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MIN'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].max().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MAX'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].mean().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MEAN'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)
tmp = invest.groupby('BTEID', as_index=0)['BTBL'].sum().rename(columns={'BTEID':'EID','BTBL':'BTBLED_SUM'})
test_invest = pd.merge(test_invest, tmp, on='EID', how='left').fillna(0)


#right features
right = right.drop_duplicates()
right_feat = right.groupby('EID', as_index=0)['TYPECODE'].count()
right_feat.rename(columns={'TYPECODE':'RIGHT_NUM'}, inplace=True)
tmp = right.groupby(['EID', 'RIGHTTYPE'])['TYPECODE'].count().unstack().reset_index().fillna(0)
tmp.columns = [i if i == 'EID' else 'RIGHTTYPE_'+str(i) for i in tmp.columns]
right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
train_right = pd.merge(train, right_feat, on='EID', how='left').fillna(0)
test_right = pd.merge(test, right_feat, on='EID', how='left').fillna(0)
del train_right['TARGET']

#project features
project = project.drop_duplicates()
project_feat = project.groupby('EID', as_index=0)['TYPECODE'].count()
project_feat.rename(columns={'TYPECODE':'PROJ_NUM'}, inplace=True)
project_feat['HOMEPROJ_NUM'] = project.groupby('EID')['IFHOME'].sum().values
project_feat['NOTHOMEPROJ_NUM'] = project_feat.PROJ_NUM - project_feat.HOMEPROJ_NUM
project_feat['NOTHOMEPROJ_RATE'] = project_feat.NOTHOMEPROJ_NUM / project_feat.PROJ_NUM
train_project = pd.merge(train, project_feat, on='EID', how='left').fillna(0)
test_project = pd.merge(test, project_feat, on='EID', how='left').fillna(0)
del train_project['TARGET']


#lawsuit features
lawsuit = lawsuit.drop_duplicates()
lawsuit_feat = lawsuit.groupby('EID', as_index=0)['TYPECODE'].count()
lawsuit_feat['LAWAMOUNT_MIN'] = lawsuit.groupby('EID')['LAWAMOUNT'].min().values
lawsuit_feat['LAWAMOUNT_MAX'] = lawsuit.groupby('EID')['LAWAMOUNT'].max().values
lawsuit_feat['LAWAMOUNT_MEAN'] = lawsuit.groupby('EID')['LAWAMOUNT'].mean().values
lawsuit_feat['LAWAMOUNT_SUM'] = lawsuit.groupby('EID')['LAWAMOUNT'].sum().values
train_lawsuit = pd.merge(train, lawsuit_feat, on='EID', how='left').fillna(0)
test_lawsuit = pd.merge(test, lawsuit_feat, on='EID', how='left').fillna(0)
del train_lawsuit['TARGET']


#breakfaith features
breakfaith_feat = breakfaith.groupby('EID', as_index=0)['TYPECODE', 'SXENDDATE'].count()
breakfaith_feat.columns = ['EID', 'SX_NUM', 'SXEND_NUM']
breakfaith_feat['SXNOTEND_NUM'] = breakfaith_feat.SX_NUM - breakfaith_feat.SXEND_NUM
breakfaith_feat['SXNOTEND_RATE'] = breakfaith_feat.SXNOTEND_NUM / breakfaith_feat.SX_NUM
train_breakfaith = pd.merge(train, breakfaith_feat, on='EID', how='left').fillna(0)
test_breakfaith = pd.merge(test, breakfaith_feat, on='EID', how='left').fillna(0)
del train_breakfaith['TARGET']


#recruit features
recruit_feat = recruit.groupby('EID', as_index=0)['RECRNUM'].sum()
recruit_feat.columns = ['EID', 'RECRSUM']
tmp = recruit.groupby(['EID', 'WZCODE'])['RECRNUM'].sum().unstack().reset_index().fillna(0)
recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
recruit['RECRMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(recruit.RECDATE)).dt.days // 30
recruit_feat['RECRMONDELMIN'] = recruit.groupby('EID')['RECRMONDEL'].min().values
recruit_feat['RECRMONDELMAX'] = recruit.groupby('EID')['RECRMONDEL'].max().values
recruit_feat['RECRMONDELMEAN'] = recruit.groupby('EID')['RECRMONDEL'].mean().values
train_recruit = pd.merge(train, recruit_feat, on='EID', how='left').fillna(0)
test_recruit = pd.merge(test, recruit_feat, on='EID', how='left').fillna(0)
del train_recruit['TARGET']


train_feat = pd.merge(train_entbase, train_alter, on='EID', how='left')
train_feat = pd.merge(train_feat, train_branch, on='EID', how='left')
train_feat = pd.merge(train_feat, train_invest, on='EID', how='left')
train_feat = pd.merge(train_feat, train_right, on='EID', how='left')
train_feat = pd.merge(train_feat, train_project, on='EID', how='left')
train_feat = pd.merge(train_feat, train_lawsuit, on='EID', how='left')
train_feat = pd.merge(train_feat, train_breakfaith, on='EID', how='left')
train_feat = pd.merge(train_feat, train_recruit, on='EID', how='left')

test_feat = pd.merge(test_entbase, test_alter, on='EID', how='left')
test_feat = pd.merge(test_feat, test_branch, on='EID', how='left')
test_feat = pd.merge(test_feat, test_invest, on='EID', how='left')
test_feat = pd.merge(test_feat, test_right, on='EID', how='left')
test_feat = pd.merge(test_feat, test_project, on='EID', how='left')
test_feat = pd.merge(test_feat, test_lawsuit, on='EID', how='left')
test_feat = pd.merge(test_feat, test_breakfaith, on='EID', how='left')
test_feat = pd.merge(test_feat, test_recruit, on='EID', how='left')


params={
#     'scale_pos_weight': 1,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'stratified':True,
    'max_depth':6,
    'min_child_weight':0.5,
    #'gamma':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    
#     'lambda':0.001,   #550
#     'alpha':0.00001,
#     'lambda_bias':0.1,
    'threads':512,
    'eta': 0.02,
    'seed':42,   
#     'silent':1
}

print train_feat.shape
print test_feat.shape
dtrain = xgb.DMatrix(train_feat, label=train.TARGET.values)
dtest = xgb.DMatrix(test_feat)

# watchlist = [(d_train, 'train'), (d_valid, 'valid')]

rounds = 5000
folds = 5
num_round = rounds
print 'run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds) 
res = xgb.cv(params, dtrain, num_round, nfold = folds, early_stopping_rounds=200, verbose_eval=2)


num_round = 1200
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=2)

ptest = model.predict(dtest)
sub = pd.DataFrame({'EID':test.EID, 'FORTARGET':[1 if i > 0.5 else 0 for i in ptest], 'PROB':ptest})
sub.to_csv('/data/BDCI2017/riskPrediction/baseline69989.csv', index=0)