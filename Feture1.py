
# coding: utf-8

# In[1]:

# In[2]:

import numpy as np
import pandas as pd
import xgboost as xgb
import copy
import re
import math
from dateutil import parser
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")#忽略错误
# reading data

DATA_PATH = 'data/'
FEATURE_PATH = 'feature/'

train = pd.read_csv(DATA_PATH+'BDCI2017/train.csv')
test = pd.read_csv(DATA_PATH+'BCI2017/evaluation_public.csv')
entbase = pd.read_csv(DATA_PATH+'/BDCI2017/1entbase.csv')
alter = pd.read_csv(DATA_PATH+'BDCI2017/2alter.csv')
branch = pd.read_csv(DATA_PATH+'BDCI2017/3branch.csv')
invest = pd.read_csv(DATA_PATH+'BDCI2017/4invest.csv')
right = pd.read_csv(DATA_PATH+'BDCI2017/5right.csv')
project = pd.read_csv(DATA_PATH+'BDCI2017/6project.csv')
lawsuit = pd.read_csv(DATA_PATH+'BDCI2017/7lawsuit.csv')
breakfaith = pd.read_csv(DATA_PATH+'BDCI2017/8breakfaith.csv')
recruit = pd.read_csv(DATA_PATH+'BDCI2017/9recruit.csv')
qualification = pd.read_csv(DATA_PATH+'BDCI2017/10qualification.csv')


# In[3]:

#填充ZCZB
def fillZCZB(r,grouped_mean):
    if (r['ZCZB']==-1):
        tmp=grouped_mean.loc[(grouped_mean.PROV==r['PROV'])&(grouped_mean.HY==r['HY'])&(grouped_mean.ETYPE==r['ETYPE'])]['ZCZB']
        return grouped_mean.iloc[tmp.index[0]]['ZCZB']
    else:
        return r['ZCZB']

def getEntbaseFeature(entbase):
    entbase_feat=copy.deepcopy(entbase)
    #ZCZB空值填充
    grouped_mean=entbase.groupby(['PROV','HY','ETYPE'],as_index=0)['ZCZB'].median()
    entbase_feat.ZCZB=entbase_feat.ZCZB.fillna(-1)
    entbase_feat.ZCZB=entbase_feat.apply(lambda r:fillZCZB(r,grouped_mean),axis=1)
    #统计身份信息的空值
    entbase_feat=entbase_feat.fillna(-1)
    entbase_feat['IDENTITY_NULL']=(entbase_feat<0).sum(axis=1)
    entbase_feat.replace(-1,0,inplace=True)
    #不同企业类型的注册资本均值
    df3=entbase.groupby('ETYPE',as_index=0)['ZCZB'].mean().rename(columns={'ZCZB':'ZCZBMEAN_BYTYPE'})
    entbase_feat=pd.merge(entbase_feat,df3,on='ETYPE',how='left')
    #注册资本与均值之差
    entbase_feat['ZCZB_DEL_ZCZBMEAN_BYTYPE']=entbase_feat.ZCZB-entbase_feat.ZCZBMEAN_BYTYPE
    #交叉
    entbase_feat['ZCZB_DEL_FINZB']=entbase_feat.ZCZB-entbase_feat.FINZB
    #特征交叉
    entbase_feat['MPNUM2+INUM2']=entbase_feat['MPNUM']*entbase_feat['MPNUM']+entbase_feat['INUM']*entbase_feat['INUM']
    entbase_feat['INUM2+FSTINUM2']=entbase_feat['INUM']*entbase_feat['INUM']+entbase_feat['FSTINUM']*entbase_feat['FSTINUM']
    entbase_feat['FSTINUM2+MPNUM2']=entbase_feat['FSTINUM']*entbase_feat['FSTINUM']+entbase_feat['MPNUM']*entbase_feat['MPNUM']
    
    tmp=entbase.groupby('HY',as_index=0)['ZCZB'].mean().rename(columns={'ZCZB':'ZCZBMEAN_BYHY'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='HY',how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEAN_BYHY']=entbase_feat.ZCZB-entbase_feat.ZCZBMEAN_BYHY

    
    tmp=entbase.groupby('HY',as_index=0)['RGYEAR'].mean().rename(columns={'RGYEAR':'RGYEARMEAN_BYHY'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='HY',how='left')
    entbase_feat['RGYEAR_DEL_RGYEARMEAN_BYHY']=entbase_feat.RGYEAR-entbase_feat.RGYEARMEAN_BYHY

    tmp=entbase.groupby('ETYPE',as_index=0)['RGYEAR'].mean().rename(columns={'RGYEAR':'RGYEARMEAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['RGYEAR_DEL_RGYEARMEAN_BYETYPE']=entbase_feat.RGYEAR-entbase_feat.RGYEARMEAN_BYETYPE
    
    tmp=entbase.groupby('HY',as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BYHY'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='HY',how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BYHY']=entbase_feat.ZCZB-entbase_feat.ZCZBMEDIAN_BYHY

        
    tmp=entbase.groupby('ETYPE',as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BYETYPE']=entbase_feat.ZCZB-entbase_feat.ZCZBMEDIAN_BYETYPE
    
    
    tmp=entbase.groupby('RGYEAR',as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BYRGYEAR'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='RGYEAR',how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BYRGYEAR']=entbase_feat.ZCZB-entbase_feat.ZCZBMEDIAN_BYRGYEAR
    
    tmp=entbase.groupby('HY',as_index=0)['INUM'].mean().rename(columns={'INUM':'INUMMEAN_BYHY'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='HY',how='left')
    entbase_feat['MPNUM_DEL_INUMMEAN_BYHY']=entbase_feat.INUM-entbase_feat.INUMMEAN_BYHY


    tmp=entbase.groupby('ETYPE',as_index=0)['INUM'].mean().rename(columns={'INUM':'INUMMEAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['MPNUM_DEL_INUMMEAN_BYETYPE']=entbase_feat.INUM-entbase_feat.INUMMEAN_BYETYPE

  
    tmp=entbase.groupby('ETYPE',as_index=0)['FSTINUM'].median().rename(columns={'FSTINUM':'FSTINUMMEDIAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['FSTINUM_DEL_FSTINUMMEDIAN_BYETYPE']=entbase_feat.FSTINUM-entbase_feat.FSTINUMMEDIAN_BYETYPE

    
    tmp=entbase.groupby('HY',as_index=0)['FSTINUM'].mean().rename(columns={'FSTINUM':'FSTINUMMEAN_BYHY'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='HY',how='left')
    entbase_feat['FSTINUM_DEL_FSTINUMMEAN_BYHY']=entbase_feat.FSTINUM-entbase_feat.FSTINUMMEAN_BYHY


    tmp=entbase.groupby('ETYPE',as_index=0)['TZINUM'].mean().rename(columns={'TZINUM':'TZINUMMEAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['TZINUM_DEL_TZINUMMEAN_BYETYPE']=entbase_feat.TZINUM-entbase_feat.TZINUMMEAN_BYETYPE

    
    tmp=entbase.groupby('ETYPE',as_index=0)['TZINUM'].median().rename(columns={'TZINUM':'TZINUMMEDIAN_BYETYPE'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='ETYPE',how='left')
    entbase_feat['TZINUM_DEL_TZINUMMEDIAN_BYETYPE']=entbase_feat.TZINUM-entbase_feat.TZINUMMEDIAN_BYETYPE


    tmp=entbase.groupby('RGYEAR',as_index=0)['INUM'].median().rename(columns={'INUM':'INUMMEDIAN_BYRGYEAR'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='RGYEAR',how='left')
    entbase_feat['INUM_DEL_INUMMEDIAN_BYRGYEAR']=entbase_feat.INUM-entbase_feat.INUMMEDIAN_BYRGYEAR
    
    tmp=entbase.groupby('RGYEAR',as_index=0)['FINZB'].median().rename(columns={'FINZB':'FINZBMEDIAN_BYRGYEAR'})
    entbase_feat=pd.merge(entbase_feat,tmp,on='RGYEAR',how='left')
    entbase_feat['FINZB_DEL_FINZBMEDIAN_BYRGYEAR']=entbase_feat.FINZB-entbase_feat.FINZBMEDIAN_BYRGYEAR

    entbase_feat['EID_NUMBER']=entbase.EID.apply(lambda x: x[1:]).astype(int)
    
    tmp=entbase.groupby(['PROV','HY'],as_index=0)['RGYEAR'].mean().rename(columns={'RGYEAR':'RGYEARMEAN_BY(PROV,HY)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','HY'],how='left')
    entbase_feat['RGYEAR_DEL_RGYEARMEAN_BY(PROV,HY)']=entbase_feat.RGYEAR-entbase_feat['RGYEARMEAN_BY(PROV,HY)']
    
    tmp=entbase.groupby(['PROV','ETYPE'],as_index=0)['RGYEAR'].mean().rename(columns={'RGYEAR':'RGYEARMEAN_BY(PROV,ETYPE)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','ETYPE'],how='left')
    entbase_feat['RGYEAR_DEL_RGYEARMEAN_BY(PROV,ETYPE)']=entbase_feat.RGYEAR-entbase_feat['RGYEARMEAN_BY(PROV,ETYPE)']

    
    tmp=entbase.groupby(['PROV','HY'],as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BY(PROV,HY)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','HY'],how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BY(PROV,HY)']=entbase_feat.ZCZB-entbase_feat['ZCZBMEDIAN_BY(PROV,HY)']

    
    tmp=entbase.groupby(['PROV','ETYPE'],as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BY(PROV,ETYPE)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','ETYPE'],how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BY(PROV,ETYPE)']=entbase_feat.ZCZB-entbase_feat['ZCZBMEDIAN_BY(PROV,ETYPE)']

    
    tmp=entbase.groupby(['PROV','RGYEAR'],as_index=0)['ZCZB'].median().rename(columns={'ZCZB':'ZCZBMEDIAN_BY(PROV,RGYEAR)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','RGYEAR'],how='left')
    entbase_feat['ZCZB_DEL_ZCZBMEDIAN_BY(PROV,RGYEAR)']=entbase_feat.ZCZB-entbase_feat['ZCZBMEDIAN_BY(PROV,RGYEAR)']

    
    tmp=entbase.groupby(['PROV','HY'],as_index=0)['INUM'].mean().rename(columns={'INUM':'INUMMEAN_BY(PROV,HY)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','HY'],how='left')
    entbase_feat['MPNUM_DEL_INUMMEAN_BY(PROV,HY)']=entbase_feat.INUM-entbase_feat['INUMMEAN_BY(PROV,HY)']

    
    tmp=entbase.groupby(['PROV','ETYPE'],as_index=0)['INUM'].mean().rename(columns={'INUM':'INUMMEAN_BY(PROV,ETYPE)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','ETYPE'],how='left')
    entbase_feat['MPNUM_DEL_INUMMEAN_BY(PROV,ETYPE)']=entbase_feat.INUM-entbase_feat['INUMMEAN_BY(PROV,ETYPE)']

    
    tmp=entbase.groupby(['PROV','ETYPE'],as_index=0)['FSTINUM'].median().rename(columns={'FSTINUM':'FSTINUMMEDIAN_BY(PROV,ETYPE)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','ETYPE'],how='left')
    entbase_feat['FSTINUM_DEL_FSTINUMMEDIAN_BY(PROV,ETYPE)']=entbase_feat.FSTINUM-entbase_feat['FSTINUMMEDIAN_BY(PROV,ETYPE)']


    tmp=entbase.groupby(['PROV','HY','ETYPE'],as_index=0)['RGYEAR'].mean().rename(columns={'RGYEAR':'RGYEARMEAN_BY(PROV,HY,ETYPE)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','HY','ETYPE'],how='left')
    entbase_feat['RGYEAR_DEL_RGYEARMEAN_BY(PROV,HY,ETYPE)']=entbase_feat.RGYEAR-entbase_feat['RGYEARMEAN_BY(PROV,HY,ETYPE)']

    
    tmp=entbase.groupby(['PROV','RGYEAR'],as_index=0)['ENUM'].mean().rename(columns={'ENUM':'ENUMMEAN_BY(PROV,RGYEAR)'})
    entbase_feat=pd.merge(entbase_feat,tmp,on=['PROV','RGYEAR'],how='left')
    entbase_feat['ENUM_DEL_ENUMMEAN_BY(PROV,RGYEAR)']=entbase_feat.ENUM-entbase_feat['ENUMMEAN_BY(PROV,RGYEAR)']

    return entbase_feat


# In[4]:

def getAlterFeature(alter):
    
    #变更时间距2017的年数差
    alter['ALTYEARDEL'] = 2017 - pd.to_datetime(alter.ALTDATE).dt.year+1
    #变更时间距2017年8月1日的月份差
    alter['ALTMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(alter.ALTDATE)).dt.days / 30
    
    alter['ALTER_YEAR'] = alter.ALTDATE.apply(lambda x:  x[:4])
    alter['ALTER_MONTH'] = alter.ALTDATE.apply(lambda x:  x[5:7])
    alter['ALTER_YEAR'] = alter['ALTER_YEAR'].astype(int)
    alter['ALTER_MONTH'] = alter['ALTER_MONTH'].astype(int)


    #去掉单位 转换为float
    altbe = []#变更前
    altaf = []#变更后
    
    for index, i in enumerate(alter.ALTBE.fillna(-1)):
        if i == -1:
            altbe.append(-1)
        elif (i[:9] == '人民币') & (i[-6:] == '万元'):
            if i[9:-6] == 'null':
                altbe.append(-1)
            else:
                altbe.append(float(i[9:-6]))
        elif (i[:6] == '港币') & (i[-6:] == '万元'):
            if (alter.loc[index].ALTDATE)[:4] == '2013' or (alter.loc[index].ALTDATE)[:4] == '2014':
                altbe.append(0.79 * float(i[6:-6]))
            else:
                altbe.append(0.837 * float(i[6:-6]))
        elif (i[:6] == '美元') & (i[-6:] == '万元'):
            if (alter.loc[index].ALTDATE)[:4] == '2013':
                altbe.append(6.1932 * float(i[6:-6]))
            elif (alter.loc[index].ALTDATE)[:4] == '2014':
                altbe.append(6.1428 * float(i[6:-6]))
            else:
                altbe.append(6.2284 * float(i[6:-6]))    
        elif i[-12:] == '万人民币':
            altbe.append(float(i[:-12]))   
        elif i[-9:] == '万美元':
            if (alter.loc[index].ALTDATE)[:4] == '2013':
                altbe.append(6.1932 * float(i[:-9]))
            elif (alter.loc[index].ALTDATE)[:4] == '2014':
                altbe.append(6.1428 * float(i[:-9]))
            else:
                altbe.append(6.2284 * float(i[:-9]))   
        elif i[-17:] == '(单位：万元)':
            altbe.append(float(i[:-17])) 
        elif i[-9:] == '万港元':
            if (alter.loc[index].ALTDATE)[:4] == '2013' or (alter.loc[index].ALTDATE)[:4] == '2014':
                altbe.append(0.79 * float(i[:-9]))
            else:
                altbe.append(0.837 * float(i[:-9]))
        elif i[-6:] == '万元':
            altbe.append(float(i[:-6]))
        elif i[-3:] == '万':
            altbe.append(float(i[:-3]))   
        else:
            altbe.append(float(i))

    for index, i in enumerate(alter.ALTAF.fillna(-1)):
        if i == -1:
            altaf.append(-1)
        elif (i[:9] == '人民币') & (i[-6:] == '万元'):
            if i[9:-6] == 'null':
                altaf.append(-1)
            else:
                altaf.append(float(i[9:-6]))
        elif (i[:6] == '港币') & (i[-6:] == '万元'):
            if (alter.loc[index].ALTDATE)[:4] == '2013' or (alter.loc[index].ALTDATE)[:4] == '2014':
                altaf.append(0.79 * float(i[6:-6]))
            else:
                altaf.append(0.837 * float(i[6:-6]))
        elif (i[:6] == '美元') & (i[-6:] == '万元'):
            if (alter.loc[index].ALTDATE)[:4] == '2013':
                altaf.append(6.1932 * float(i[6:-6]))
            elif (alter.loc[index].ALTDATE)[:4] == '2014':
                altaf.append(6.1428 * float(i[6:-6]))
            else:
                altaf.append(6.2284 * float(i[6:-6]))    
        elif i[-12:] == '万人民币':
            altaf.append(float(i[:-12]))   
        elif i[-9:] == '万美元':
            if (alter.loc[index].ALTDATE)[:4] == '2013':
                altaf.append(6.1932 * float(i[:-9]))
            elif (alter.loc[index].ALTDATE)[:4] == '2014':
                altaf.append(6.1428 * float(i[:-9]))
            else:
                altaf.append(6.2284 * float(i[:-9]))   
        elif i[-17:] == '(单位：万元)':
            altaf.append(float(i[:-17])) 
        elif i[-9:] == '万港元':
            if (alter.loc[index].ALTDATE)[:4] == '2013' or (alter.loc[index].ALTDATE)[:4] == '2014':
                altaf.append(0.79 * float(i[:-9]))
            else:
                altaf.append(0.837 * float(i[:-9]))
        elif i[-6:] == '万元':
            altaf.append(float(i[:-6]))
        elif i[-3:] == '万':
            altaf.append(float(i[:-3]))   
        else:
            altaf.append(float(i))
        
    alter['ALTBE'] = altbe
    alter['ALTAF'] = altaf
    alter.ALTBE.replace( -1,np.nan, inplace=True)#-1用空值代替
    alter.ALTAF.replace(-1,np.nan, inplace=True)
    
    #变更数值变化
    alter['ALTBADEL'] = alter.ALTAF - alter.ALTBE
 
    #企业变更次数
    alter_feat = alter.groupby('EID', as_index=0)['ALTERNO'].count().rename(columns={'ALTERNO':'ALT_NUM'})#企业变更次数
    alter_feat.ALT_NUM=alter_feat.ALT_NUM.astype(int)
    
    #ALTBE
    tmp=alter.groupby('EID',as_index=0)['ALTBE'].min().rename(columns={'ALTBE':'ALTBE_MIN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTBE'].max().rename(columns={'ALTBE':'ALTBE_MAX'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTBE'].mean().rename(columns={'ALTBE':'ALTBE_MEAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTBE'].median().rename(columns={'ALTBE':'ALTBE_MEDIAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')


    #ALTAF
    tmp=alter.groupby('EID',as_index=0)['ALTAF'].min().rename(columns={'ALTAF':'ALTAF_MIN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTAF'].max().rename(columns={'ALTAF':'ALTAF_MAX'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTAF'].mean().rename(columns={'ALTAF':'ALTAF_MEAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTAF'].median().rename(columns={'ALTAF':'ALTAF_MEDIAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left') 
    tmp=alter.groupby('EID',as_index=0)['ALTAF'].count().rename(columns={'ALTAF':'ALTAF_COUNT'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')

    
    #ALTYEARDEL
    tmp=alter.groupby('EID',as_index=0)['ALTYEARDEL'].min().rename(columns={'ALTYEARDEL':'ALTYEARDEL_MIN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTYEARDEL'].max().rename(columns={'ALTYEARDEL':'ALTYEARDEL_MAX'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTYEARDEL'].mean().rename(columns={'ALTYEARDEL':'ALTYEARDEL_MEAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    
    #ALTMONDEL
    tmp=alter.groupby('EID',as_index=0)['ALTMONDEL'].min().rename(columns={'ALTMONDEL':'ALTMONDEL_MIN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTMONDEL'].max().rename(columns={'ALTMONDEL':'ALTMONDEL_MAX'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTMONDEL'].mean().rename(columns={'ALTMONDEL':'ALTMONDEL_MEAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    
    
    #ALTBADEL
    tmp=alter.groupby('EID',as_index=0)['ALTBADEL'].min().rename(columns={'ALTBADEL':'ALTBADEL_MIN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTBADEL'].max().rename(columns={'ALTBADEL':'ALTBADEL_MAX'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    tmp=alter.groupby('EID',as_index=0)['ALTBADEL'].mean().rename(columns={'ALTBADEL':'ALTBADEL_MEAN'})
    alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
    
    
    #ALTERNO类型特征的数量{'01', '02', '03', '04', '05', '10', '12', '13', '14', '27', '99', 'A_015'}
    for i in set(alter.ALTERNO):
        tmp=alter[alter.ALTERNO==i].groupby('EID').size().reset_index().rename(columns = {0:'ALTER'+i+'_NUM'})
        alter_feat=pd.merge(alter_feat,tmp,on='EID',how='left')
        
    alter_feat['05NUM_RATE']=alter_feat.ALTER05_NUM / alter_feat.ALT_NUM
    alter_feat['27NUM_RATE']=alter_feat.ALTER27_NUM / alter_feat.ALT_NUM
    
    return alter_feat


# In[5]:

def getBranchFeature(branch):
    #分支存活时间 endyear为空时为空
    branch['B_YEARGAP'] = branch.B_ENDYEAR - branch.B_REYEAR
    #成立年距2017的年数
    branch['B_REYEARDEL']=2017-branch.B_REYEAR+1
    #关停年距2017的年数
    branch['B_ENDYEAR']=2017-branch.B_ENDYEAR+1
    #企业分支数
    branch_feat = branch.groupby('EID', as_index=0)['TYPECODE'].count().rename(columns={'TYPECODE':'BRANCH_NUM'})
    #企业同省分支数
    tmp = branch.groupby('EID', as_index=0)['IFHOME'].sum().rename(columns={'IFHOME':'B_IFHOME_NUM'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    #企业同省分支率
    branch_feat['B_IFHOME_RATE']=branch_feat.B_IFHOME_NUM/branch_feat.BRANCH_NUM
    #分支关闭的数量
    tmp = branch.groupby('EID', as_index=0)['B_ENDYEAR'].count().rename(columns={'B_ENDYEAR':'B_CLOSE_NUM'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    #分支关闭率
    branch_feat['B_CLOSE_RATE'] = branch_feat.B_CLOSE_NUM / branch_feat.BRANCH_NUM
    #分支存活的数量
    branch_feat['B_LIVE_NUM']=branch_feat.BRANCH_NUM-branch_feat['B_CLOSE_NUM']
    #分支存活率
    branch_feat['B_LIVE_RATE']=branch_feat['B_LIVE_NUM']/ branch_feat.BRANCH_NUM
    #统计量
    #B_REYEAR
    tmp=branch.groupby('EID', as_index=0)['B_REYEARDEL'].min().rename(columns={'B_REYEARDEL':'B_REYEARMIN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_REYEARDEL'].max().rename(columns={'B_REYEARDEL':'B_REYEARMAX'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_REYEARDEL'].mean().rename(columns={'B_REYEARDEL':'B_REYEARMEAN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    
    #B_ENDYEAR
    tmp=branch.groupby('EID', as_index=0)['B_ENDYEAR'].min().rename(columns={'B_ENDYEAR':'B_ENDYEARMIN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_ENDYEAR'].max().rename(columns={'B_ENDYEAR':'B_ENDYEARMAX'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_ENDYEAR'].mean().rename(columns={'B_ENDYEAR':'B_ENDYEARMEAN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    #B_YEARGAP
    tmp=branch.groupby('EID', as_index=0)['B_YEARGAP'].min().rename(columns={'B_YEARGAP':'B_YEARGAPMIN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_YEARGAP'].max().rename(columns={'B_YEARGAP':'B_YEARGAPMAX'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')
    tmp=branch.groupby('EID', as_index=0)['B_YEARGAP'].mean().rename(columns={'B_YEARGAP':'B_YEARGAPMEAN'})
    branch_feat=pd.merge(branch_feat,tmp,on='EID',how='left')

    return branch_feat


# In[6]:

def getInvestFeature(invest):
    invest = invest.drop_duplicates()
    #投资的个数
    invest_feat=invest.groupby('EID', as_index=0)['BTEID'].count().rename(columns={'BTEID':'INVEST_NUM'})
    #被投的个数
    tmp = invest.groupby('BTEID', as_index=0)['EID'].count().rename(columns={'BTEID':'EID', 'EID':'BT_NUM'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    #投资企业失败的个数（空才是正常营业）count可以过滤nan
    tmp = invest.groupby('EID', as_index=0)['BTENDYEAR'].count().rename(columns={'BTENDYEAR':'INVESTFAIL_NUM'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    #投资企业的失败率
    invest_feat['INVESTFAIL_RATE'] = invest_feat.INVESTFAIL_NUM / invest_feat.INVEST_NUM
    #持股比例统计量
    tmp = invest.groupby('EID', as_index=0)['BTBL'].min().rename(columns={'BTBL':'BTBL_MIN'})#持股比例
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('EID', as_index=0)['BTBL'].max().rename(columns={'BTBL':'BTBL_MAX'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('EID', as_index=0)['BTBL'].mean().rename(columns={'BTBL':'BTBL_MEAN'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('EID', as_index=0)['BTBL'].sum().rename(columns={'BTBL':'BTBL_SUM'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    
    #被别企持股的比例
    tmp = invest.groupby('BTEID', as_index=0)['BTBL'].min().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MIN'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('BTEID', as_index=0)['BTBL'].max().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MAX'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('BTEID', as_index=0)['BTBL'].mean().rename(columns={'BTEID':'EID','BTBL':'BTBLED_MEAN'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    tmp = invest.groupby('BTEID', as_index=0)['BTBL'].sum().rename(columns={'BTEID':'EID','BTBL':'BTBLED_SUM'})
    invest_feat = pd.merge(invest_feat, tmp, on='EID', how='left')
    
    return invest_feat


# In[7]:

def getRightFeature(right):
    
    #ASKDATE差距的月份
    right['RIGHT_ASKMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(right.ASKDATE)).dt.days / 30

    #FBDATE差距的月份
    right['RIGHT_FBMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(right.FBDATE)).dt.days / 30

    #权利个数
    right_feat = right.groupby('EID', as_index=0)['TYPECODE'].count().rename(columns={'TYPECODE':'RIGHT_NUM'})
    #统计某企业某权利类型下的权利个数{11, 12, 20, 30, 40, 50, 60}
    tmp = right.groupby(['EID', 'RIGHTTYPE'])['TYPECODE'].count().unstack().reset_index()#.fillna(0)
    tmp.columns = [i if i == 'EID' else 'RIGHTTYPE_'+str(i)+'NUM' for i in tmp.columns]
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    #RIGHT_ASKMONDEL
    tmp = right.groupby('EID', as_index=0)['RIGHT_ASKMONDEL'].min().rename(columns={'RIGHT_ASKMONDEL':'RIGHT_ASKMONDELMIN'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    tmp = right.groupby('EID', as_index=0)['RIGHT_ASKMONDEL'].max().rename(columns={'RIGHT_ASKMONDEL':'RIGHT_ASKMONDELMAX'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    tmp = right.groupby('EID', as_index=0)['RIGHT_ASKMONDEL'].mean().rename(columns={'RIGHT_ASKMONDEL':'RIGHT_ASKMONDELMEAN'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    
    #RIGHT_FBMONDEL
    tmp = right.groupby('EID', as_index=0)['RIGHT_FBMONDEL'].min().rename(columns={'RIGHT_FBMONDEL':'RIGHT_FBMONDELMIN'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    tmp = right.groupby('EID', as_index=0)['RIGHT_FBMONDEL'].max().rename(columns={'RIGHT_FBMONDEL':'RIGHT_FBMONDELMAX'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    tmp = right.groupby('EID', as_index=0)['RIGHT_FBMONDEL'].mean().rename(columns={'RIGHT_FBMONDEL':'RIGHT_FBMONDELMEAN'})
    right_feat = pd.merge(right_feat, tmp, on='EID', how='left')
    
    #FBDATE非空值个数 申请未通过
    tmp=right.groupby('EID',as_index=0)['FBDATE'].count().rename(columns={'FBDATE':'FBNUM'})
    right_feat=pd.merge(right_feat,tmp,on='EID',how='left')
    
    #FBDATE非空值概率
    right_feat['FBNUMRATE']=right_feat.FBNUM/right_feat.RIGHT_NUM 

    #TYPECODE取其字母做类型特征，统计每类型的个数
    def getchar(s):
        if s.isdigit()==False:
            return s[0:3]
        else:
            return 'oth'
        
    right['TYPECODE_CHAR']=right.TYPECODE.apply(lambda x:getchar(x))
    col=set(right.TYPECODE_CHAR)
    for i in col:
        df= right[right.TYPECODE_CHAR==i].groupby(['EID']).size().reset_index().rename(columns = {0:'TYPECODE_CHAR'+i+'_NUM'})
        right_feat=pd.merge(right_feat,df,on='EID',how='left').fillna(0)
    #1029add offline rise online low
    #RIGHTTYPE40 50 60 个数的占比    
    right_feat['RIGHTTYPE_40RATE']=right_feat['RIGHTTYPE_40NUM']/right_feat['RIGHT_NUM']
    right_feat['RIGHTTYPE_50RATE']=right_feat['RIGHTTYPE_50NUM']/right_feat['RIGHT_NUM']
    right_feat['RIGHTTYPE_60RATE']=right_feat['RIGHTTYPE_60NUM']/right_feat['RIGHT_NUM']

    return right_feat


# In[8]:

def getProjectFeature(project):
    #项目个数
    project_feat = project.groupby('EID', as_index=0)['TYPECODE'].count().rename(columns={'TYPECODE':'PROJ_NUM'})
    #同省的项目个数
    tmp=project.groupby('EID', as_index=0)['IFHOME'].sum().rename(columns={'IFHOME':'HOMEPROJ_NUM'})
    project_feat = pd.merge(project_feat, tmp, on='EID', how='left')
    #外省的项目个数
    project_feat['NOTHOMEPROJ_NUM'] = project_feat.PROJ_NUM - project_feat.HOMEPROJ_NUM
    #外省的项目率
    project_feat['NOTHOMEPROJ_RATE'] = project_feat.NOTHOMEPROJ_NUM / project_feat.PROJ_NUM
    return project_feat


# In[9]:

def getLawsuitFeature(lawsuit):
    #案件个数
    lawsuit_feat = lawsuit.groupby('EID', as_index=0)['TYPECODE'].count().rename(columns={'TYPECODE':'LAWSUIT_NUM'})
    #金额案件
    tmp=lawsuit.groupby('EID', as_index=0)['LAWAMOUNT'].min().rename(columns={'LAWAMOUNT':'LAWAMOUNT_MIN'})
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on='EID', how='left')
    tmp=lawsuit.groupby('EID', as_index=0)['LAWAMOUNT'].max().rename(columns={'LAWAMOUNT':'LAWAMOUNT_MAX'})
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on='EID', how='left')
    tmp=lawsuit.groupby('EID', as_index=0)['LAWAMOUNT'].mean().rename(columns={'LAWAMOUNT':'LAWAMOUNT_MEAN'})
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on='EID', how='left')
    tmp=lawsuit.groupby('EID', as_index=0)['LAWAMOUNT'].sum().rename(columns={'LAWAMOUNT':'LAWAMOUNT_SUM'})
    lawsuit_feat = pd.merge(lawsuit_feat, tmp, on='EID', how='left')
    return lawsuit_feat


# In[10]:

def getBreakfaithFeature(breakfaith):
    breakfaith_feat = breakfaith.groupby('EID', as_index=0)['TYPECODE', 'SXENDDATE'].count()#失信个数和失信结束的个数
    breakfaith_feat.columns = ['EID', 'SX_NUM', 'SXEND_NUM']#修改列名
    breakfaith_feat['SXNOTEND_NUM'] = breakfaith_feat.SX_NUM - breakfaith_feat.SXEND_NUM#失信未结束的个数
    breakfaith_feat['SXNOTEND_RATE'] = breakfaith_feat.SXNOTEND_NUM / breakfaith_feat.SX_NUM#失信未结束率

    return breakfaith_feat


# In[11]:

def cleanPNUM(df):
    data = df['PNUM'].fillna('0').values
    pnum = []
    for row in data:
        if '人' in row:
            #print row
            row = int(row[:-3])
        elif row!='若干':
            row = int(row)
        elif row=='若干':
            row = 4.7
        pnum.append(row)

    df['PNUM'] = pnum
    return df

def getRecruitFeature(recruit):
    recruit = cleanPNUM(recruit)
    recruit['RECDATE_YEAR'] = recruit.RECDATE.apply(lambda x:  x[:4])
    recruit['RECDATE_MONTH'] = recruit.RECDATE.apply(lambda x:  x[5:7])
    recruit['RECDATE_YEAR'] = recruit['RECDATE_YEAR'].astype(int)
    recruit['RECDATE_MONTH'] = recruit['RECDATE_MONTH'].astype(int)
    recruit['RECRMONDEL'] = (datetime(2017, 8, 1) - pd.to_datetime(recruit.RECDATE)).dt.days / 30#差距的月份
    #招聘职位的总数
    recruit_feat = recruit.groupby('EID', as_index=0)['PNUM'].sum().rename(columns={'PNUM':'PSUM'})
    #2015年8月招聘的数量与企业平均招聘数量之差
    tmp=recruit.groupby('EID', as_index=0)['PNUM'].mean().rename(columns={'PNUM':'PNUM_MEAN'})
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
      
    df=recruit[pd.to_datetime(recruit.RECDATE)=='2015-08-01']
    tmp=df.groupby('EID', as_index=0)['PNUM'].sum().rename(columns={'PNUM':'201508_PSUM'}).fillna(0)
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left').fillna(0)
    recruit_feat['201508_DELMEAN']=recruit_feat['201508_PSUM']-recruit_feat['PNUM_MEAN']

    recruit_feat.drop(['201508_PSUM','PNUM_MEAN'],axis=1,inplace=True)
   
    #某企业在某招聘网站上招聘的人数
    tmp = recruit.groupby(['EID', 'WZCODE'])['PNUM'].sum().unstack().reset_index().fillna(0)#某企业在某招聘网站上招聘的职位总数
    tmp.columns  = ['EID','ZP01_PSUM','ZP02_PSUM','ZP03_PSUM']
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left') 
    #RECRMONDEL
    tmp=recruit.groupby('EID', as_index=0)['RECRMONDEL'].min().rename(columns={'RECRMONDEL':'RECRMONDELMIN'})
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
    tmp=recruit.groupby('EID', as_index=0)['RECRMONDEL'].max().rename(columns={'RECRMONDEL':'RECRMONDELMAX'})
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
    tmp=recruit.groupby('EID', as_index=0)['RECRMONDEL'].mean().rename(columns={'RECRMONDEL':'RECRMONDELMEAN'})
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left')
   
    #某企业在某招聘网站上招聘的职位总数
    tmp = recruit.groupby(['EID', 'WZCODE'])['POSCODE'].count().unstack().reset_index()
    tmp.columns  = ['EID','ZP01_POSCODE_NUM','ZP02_POSCODE_NUM','ZP03_POSCODE_NUM']
    recruit_feat = pd.merge(recruit_feat, tmp, on='EID', how='left') 
    
    return recruit_feat


# In[12]:

from datetime import datetime

def toDatetime(date):
    if date != -1:
        year = int(date[:4])
        month = int(date[6:-2])
        return datetime(year,month,1)
    else:
        return np.nan

def getQualificationFeature(qualification):
    qualification.BEGINDATE = qualification.BEGINDATE.apply(lambda x: toDatetime(x))
    qualification.EXPIRYDATE = qualification.EXPIRYDATE.fillna(-1)
    qualification.EXPIRYDATE = qualification.EXPIRYDATE.apply(lambda x: toDatetime(x))
    qualification.EXPIRYDATE.replace(-1,np.nan,inplace=True)

    qualification['BEGINDATE_MONDEL'] = (datetime(2017, 8, 1) - qualification.BEGINDATE).dt.days / 30#差距的月份
    qualification['EXPIRYDATE_MONDEL'] = (datetime(2017, 8, 1) - qualification.EXPIRYDATE).dt.days / 30
    qualification['EXPIRYDATE_DEL_BEGINDATE'] = (qualification.EXPIRYDATE - qualification.BEGINDATE).dt.days / 30

    qualification_feat = qualification.groupby('EID',as_index=0)['ADDTYPE'].count().rename(columns={'ADDTYPE':'ADDTYPE_NUM'})
    
    tmp=qualification.groupby('EID', as_index=0)['BEGINDATE_MONDEL'].min().rename(columns={'BEGINDATE_MONDEL':'BEGINDATE_MIN'})
    qualification_feat=pd.merge(qualification_feat,tmp,on='EID',how='left')
    tmp=qualification.groupby('EID', as_index=0)['BEGINDATE_MONDEL'].max().rename(columns={'BEGINDATE_MONDEL':'BEGINDATE_MAX'})
    qualification_feat=pd.merge(qualification_feat,tmp,on='EID',how='left')
    tmp=qualification.groupby('EID', as_index=0)['BEGINDATE_MONDEL'].mean().rename(columns={'BEGINDATE_MONDEL':'BEGINDATE_MEAN'})
    qualification_feat=pd.merge(qualification_feat,tmp,on='EID',how='left')

    tmp = qualification.groupby('EID',as_index=0)['ADDTYPE'].sum().rename(columns={'ADDTYPE':'ADDTYPE_SUM'})
    qualification_feat=pd.merge(qualification_feat,tmp,on='EID',how='left')
    
    return qualification_feat


# In[13]:

entbase_feat=getEntbaseFeature(entbase)
alter_feat = getAlterFeature(alter)
branch_feat = getBranchFeature(branch)
invest_feat = getInvestFeature(invest)
right_feat = getRightFeature(right)
project_feat = getProjectFeature(project)
lawsuit_feat = getLawsuitFeature(lawsuit)
breakfaith_feat = getBreakfaithFeature(breakfaith)
recruit_feat = getRecruitFeature(recruit)
qualification_feat = getQualificationFeature(qualification)


# In[14]:

feature = pd.merge(entbase_feat,alter_feat,on='EID',how='left')
feature = pd.merge(feature,branch_feat,on='EID',how='left')
feature = pd.merge(feature,invest_feat,on='EID',how='left')
feature = pd.merge(feature,right_feat,on='EID',how='left')
feature = pd.merge(feature,project_feat,on='EID',how='left')
feature = pd.merge(feature,lawsuit_feat,on='EID',how='left')
feature = pd.merge(feature,breakfaith_feat,on='EID',how='left')
feature = pd.merge(feature,recruit_feat,on='EID',how='left')
feature = pd.merge(feature,qualification_feat,on='EID',how='left')


# In[15]:

trainset = pd.merge(train,feature,on='EID',how='left')
testset = pd.merge(test,feature,on='EID',how='left')
trainset = trainset.drop('ENDDATE',axis=1)


# In[16]:

x = trainset.fillna(-999)
trainset['NULL_NUM'] =(x == -999).sum(axis=1)
y = testset.fillna(-999)
testset['NULL_NUM'] =(y == -999).sum(axis=1)


# In[ ]:

trainset.to_csv(FEATURE_PATH +'trainFeature_zx_6928_withEID.csv',index=0)
testset.to_csv(FEATURE_PATH +'testFeature_zx_6928_withEID.csv',index=0)


# In[17]:

# trainFeature = trainset.drop(['EID','TARGET'],axis=1)
# trainLabel = trainset.TARGET.values
# testFeature = testset.drop('EID',axis=1)
# testIndex = testset.EID.values


# # In[18]:

# trainFeature.info()


# # In[19]:

# testFeature.info()


# # In[20]:

# import xgboost as xgb
# config = {
#     'rounds': 10000,
#     'folds': 5
# }

# params = {
#     'booster':'gbtree',
#     'objective':'binary:logistic',
#     'stratified':True,
#     'max_depth':8,
#     'min_child_weight':1,
#     'gamma':4,
#     'subsample':0.7,#0.7
#     'colsample_bytree':0.6, 
#     'lambda':3, 
#     'eta':0.02,
#     'seed':20,
#     'silent':1,
#     'eval_metric':'auc'
# }

# def xgbCV(trainFeature, trainLabel, params, rounds):

#     dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
#     params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
#     num_round = rounds
#     print 'run cv: ' + 'round: ' + str(rounds)
#     res = xgb.cv(params, dtrain, num_round, verbose_eval = 10,early_stopping_rounds=50)
#     return len(res)

# def xgbPredict(trainFeature,trainLabel,testFeature,rounds,params):
#     params['scale_pos_weights '] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
#     dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
#     dtest = xgb.DMatrix(testFeature, label = np.zeros(testFeature.shape[0]))

#     watchlist  = [(dtrain,'train')]
#     num_round = rounds
    
#     model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 50)
#     predict = model.predict(dtest)
#     return model, predict

# def storeResult(testIndex,preds,threshold,name,):
#     result = pd.DataFrame({'EID':testIndex,'FORTARGET':0,'PROB':preds})
#     mask = result['PROB'] >= threshold
#     result.at[mask,'FORTARGET'] = 1
#     result['PROB'] = result['PROB'].apply(lambda x:round(x,4))
#     result.to_csv('./data/BDCI2017/riskPrediction/'+name+'.csv',index=0)
#     return result

# iterations = xgbCV(trainFeature,trainLabel,params,config['rounds'])#3折


# In[ ]:



