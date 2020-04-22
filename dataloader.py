# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:27:48 2020

@author: Eugene
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
tqdm.pandas()


def gen_edges(target_rank, save=False):
    p_p = pd.read_pickle('paper_paper.pkl')
    p_v = pd.read_pickle('paper_venue.pkl')
    p_a = pd.read_pickle('paper_author.pkl')
    keyword_list = np.load('keyword_list.npy')

    data = pd.read_pickle('dblp_0105.pkl')
    paper_time_step = data['time_step']

    # target_rank = 162  # default 281
    target_year = 2018
    hot_rate = 5
    good_author_num = 2

    last_pv_id = np.max(p_v['new_venue_id'])
    last_pv_id += 1

    # venue year
    data2 = data.copy()
    venue_year = pd.DataFrame(data2[['venue_year','time_step']].drop_duplicates().reset_index(drop=True))
    venue_year = venue_year.sort_values(by=['time_step'])
    venue_year['new_venue_year_id'] = np.arange(last_pv_id,last_pv_id+len(venue_year))
    paper_venue_year = pd.merge(data,venue_year)[['new_papr_id','new_venue_year_id','time_step']]
    paper_venue_year = paper_venue_year.reset_index(drop=True)
    # origin_venue = p_v['new_venue_id'].copy().drop_duplicates()

    '''paper發在哪一年的哪個venue裡面，2018KDD和2019KDD是兩個不同的venue'''
    p_vy = paper_venue_year.copy()
    pvy = p_vy[['new_papr_id','new_venue_year_id']]
    paper_year = p_v[['new_papr_id','time_step','year']]
    pvy_new = pd.merge(paper_year,pvy)
    pvy_group = pvy_new.groupby('year')
    c = 8  # year_venue relation
    pvy_new = pd.DataFrame()
    for i in range(2011,2020):
        group_temp = pvy_group.get_group(i)
        group_temp = group_temp.reset_index(drop=True)
        rel = pd.DataFrame(np.ones(len(group_temp))*c)
        rel.columns = ['rel']
        group_temp = pd.concat((group_temp,rel),axis=1)
        group_temp = group_temp[['new_papr_id','new_venue_year_id','rel','time_step']]
        group_temp.columns = ['head','tail','rel','time_step']
        c += 1
        pvy_new = pvy_new.append(group_temp)

    pvy = pvy_new[pvy_new['time_step']<target_rank][['head', 'tail', 'rel']]
    print(pvy['tail'].unique().shape)

    #venue year venue
    '''EX:2018KDD和2019KDD都屬於KDD這個venue'''
    venue_year_venue = pd.merge(paper_venue_year,p_v)[['new_venue_year_id','new_venue_id','time_step']].drop_duplicates()
    venue_year_venue = venue_year_venue.reset_index(drop=True)
    last_vy_id = np.max(venue_year_venue['new_venue_year_id'])
    last_vy_id += 1
    vy_v = venue_year_venue.copy()
    vyv = vy_v[vy_v['time_step']<target_rank][['new_venue_year_id','new_venue_id']]
    vyv = vyv.reset_index(drop=True)
    rel = pd.DataFrame(np.ones(len(vyv))*7)  # venue id is 7
    vyv = pd.concat((vyv,rel),axis=1)
    vyv.columns = ['head','tail','rel']

    # keyword
    '''paper裡面有哪些keyword的關係'''
    keyword_list = list(dict.fromkeys(keyword_list))
    keydict = {keyword_list[i]: i+last_vy_id for i in range(0, len(keyword_list))}

    keyword = data['keyword']
    def get_keyword_id(df):
        key_id = []
        for i in df:
            key_id.append(keydict[i])
        return key_id
    paper_keyword = []
    for i in range(len(keyword)):
        for k in keyword.iloc[i]:
            paper_keyword.append([i,keydict[k],paper_time_step.iloc[i]])

    paper_keyword = pd.DataFrame(paper_keyword)
    paper_keyword.columns = ['new_papr_id','keyword_id','time_step']

    pk = paper_keyword[paper_keyword['time_step']<target_rank][['new_papr_id','keyword_id']]
    pk = pk.reset_index(drop=True)
    rel = pd.DataFrame(np.ones(len(pk))*6)
    pk = pd.concat((pk,rel),axis=1)
    pk.columns = ['head','tail','rel']

    #hot paper
    #target_REF = p_p[p_p['time_step']<target_rank]
    #cited_value = target_REF['new_cited_papr_id'].value_counts()
    #cited_value = pd.DataFrame(cited_value)
    #cited_value['pub_year'] = target_year - data.iloc[cited_value.index]['year'] + 1
    #cited_value['cited_rate'] = cited_value['new_cited_papr_id']/cited_value['pub_year']
    #good_paper = list(cited_value[cited_value['cited_rate']>hot_rate].index)

    #paper hot 先篩reference時間再找hot paper>0.5
    #pp = p_p[p_p['time_step']<target_rank][['new_papr_id','new_cited_papr_id']]
    #
    #def choose_hot_paper(df):
    #    if df in good_paper:
    #        return df
    #    else:
    #        return None
    #
    #print('process hot cite')
    #pp_hot = pd.DataFrame(pp['new_cited_papr_id'].progress_apply(choose_hot_paper))
    #pp_hot.columns = ['good_paper']
    #pp_hot = pd.concat((pp,pp_hot),axis=1).dropna()
    #pp_hot = pp_hot.drop(columns=['new_cited_papr_id'])
    #pp_hot = pp_hot.reset_index(drop=True)
    #pp_hot.columns = ['head','tail']
    #ph = pp_hot.copy()
    #rel = pd.DataFrame(np.ones(len(ph))*2)
    #ph = pd.concat((ph,rel),axis=1)
    #ph.columns = ['head','tail','rel']

    #delete author 先照時間篩選pa 再找寫>1篇的作者
    '''paper的author是誰的關係'''
    pa = p_a[p_a['time_step']<target_rank][['new_papr_id','new_author_id']]
    author_value = pa['new_author_id'].value_counts()
    author_value = pd.DataFrame(author_value)
    bad_author = list(author_value[author_value['new_author_id']<=good_author_num].index)
    good_author = list(author_value[author_value['new_author_id']>good_author_num].index)
    pa_new = pa[~pa['new_author_id'].isin(bad_author)]
    pa_new.columns = ['head','tail']
    pa_new = pa_new.reset_index(drop=True)

    rel = pd.DataFrame(np.ones(len(pa_new)))  # author relation is 1
    pa = pd.concat((pa_new,rel),axis=1)
    pa.columns = ['head','tail','rel']

    #self cite 會因為不同時間作者不同而改變
    #paper_author = defaultdict(list)
    #
    #for i in pa_new.values:
    #    paper_author[i[0]].append(i[1])
    #
    #exist_paper = list(paper_author.keys())
    #
    #def find_self_cite(cite,cited):
    #    if cite in exist_paper and cited in exist_paper:
    #        if len(list(set(paper_author[cite]).intersection(set(paper_author[cited]))))>0:
    #            return 1
    #        return 0
    #    else:
    #        return 0
    #
    #print('process self cite')
    #pp['self_cite'] = pp.progress_apply(lambda row: find_self_cite(row['new_papr_id'], row['new_cited_papr_id']), axis=1)
    #pp_self_cite = pp[pp['self_cite']==1]
    #pp_self_cite = pp_self_cite.drop(columns=['self_cite'])
    #pp_self_cite.columns = ['head','tail']
    #pp_self_cite = pp_self_cite.reset_index(drop=True)
    #pself = pp_self_cite.copy()
    #rel = pd.DataFrame(np.ones(len(pself))*4)
    #pself = pd.concat((pself,rel),axis=1)
    #pself.columns = ['head','tail','rel']


    #newest cite
    #pp_newest = p_p[['new_papr_id','new_cited_papr_id','year','time_step']].reset_index(drop=True)
    #pp_newest.columns = ['head','tail','year','time_step']
    #cite_year = data.iloc[pp_newest['tail'].values]['year'].reset_index(drop=True)
    #pp_newest['cite_year'] = cite_year
    #pp_newest['cite_newest'] = pp_newest['year'] - pp_newest['cite_year']
    #pp_newest_cite = pp_newest[(pp_newest['cite_newest']<=1) & (pp_newest['cite_newest']>=0)][['head','tail','time_step']]
    #pp_new = pp_newest_cite[pp_newest_cite['time_step']<target_rank].drop(columns=['time_step'])
    #pp_new = pp_new.reset_index(drop=True)
    #
    #pnew = pp_new.copy()
    #rel = pd.DataFrame(np.ones(len(pnew))*3)
    #pnew = pd.concat((pnew,rel),axis=1)
    #pnew.columns = ['head','tail','rel']


    #survey cite pp已篩過時間
    #title = pd.DataFrame(data['title'].str.lower())
    #title['if suevey or not'] = title['title'].apply(lambda x : 1 if 'survey' in x else 0)
    #title['if review or not'] = title['title'].apply(lambda x : 1 if 'review on' in x or 'review of' in x else 0)
    #survey_paper_list = pd.concat((data['new_papr_id'],title),axis=1)
    #survey_paper_list = survey_paper_list[(survey_paper_list['if review or not']==1) | (survey_paper_list['if suevey or not'] ==1)]['new_papr_id'].tolist()
    #
    #def choose_survey_paper(df):
    #    if df in survey_paper_list:
    #        return df
    #    else:
    #        return None
    #
    #print('process survey cite')
    #pp_survey = pd.DataFrame(pp['new_cited_papr_id'].progress_apply(choose_survey_paper))
    #pp_survey.columns = ['survey_paper']
    #pp_survey = pd.concat((pp,pp_survey),axis=1).dropna()
    #pp_survey = pp_survey.drop(columns=['new_cited_papr_id','self_cite'])
    #pp_survey = pp_survey.reset_index(drop=True)
    #psurvey = pp_survey.copy()
    #rel = pd.DataFrame(np.ones(len(psurvey))*5)
    #psurvey = pd.concat((psurvey,rel),axis=1)
    #psurvey.columns = ['head','tail','rel']

    #pp
    '''paper cite paper的關係'''
    pp = p_p[p_p['time_step']<target_rank-1][['new_papr_id','new_cited_papr_id']]
    pp = pp.reset_index(drop=True)
    rel = pd.DataFrame(np.zeros(len(pp)))
    pp = pd.concat((pp,rel),axis=1)
    pp.columns = ['head', 'tail', 'rel']

    # edge list
    # all_edge = pd.concat([pp,pa,ph,pnew,pself,psurvey,pk,vyv,pvy])
    all_edge = pd.concat([pp,pa,pk,vyv,pvy])

    all_edge = all_edge.drop_duplicates()
    all_edge = all_edge.reset_index(drop=True)

    if save:
        with open('all_edge_'+str(target_rank)+'.pkl', 'wb') as file:
            pickle.dump(all_edge, file)

    # map to new id
    # index = np.arange(len(list(set(pd.unique(all_edge['tail'])).union(set(pd.unique(all_edge['head']))))))
    # content = np.array(list(set(pd.unique(all_edge['tail'])).union(set(pd.unique(all_edge['head'])))))
    # new_index_dict = dict(zip(content, index))
    # all_edge['head'] = all_edge['head'].map(new_index_dict)
    # all_edge['tail'] = all_edge['tail'].map(new_index_dict)

    return all_edge

# def compute_node(el):
#     h = list(el['head'].values)
#     t = list(el['tail'].values)
#     h.extend(t)
#     h = list(set(h))
#     print('node nums:',len(h))
#     print('max node id:',int(max(h)))
#     print('edge nums:',len(el))
#     return int(max(h)),len(h)
#
# max_id,num_nodes = compute_node(all_edge)
