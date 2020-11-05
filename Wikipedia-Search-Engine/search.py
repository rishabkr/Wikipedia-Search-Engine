import nltk
from collections import defaultdict
import xml.etree.ElementTree as etree
nltk_stemmer=nltk.stem.SnowballStemmer('english')
from tqdm import tqdm
from collections import defaultdict
import sys
import operator
from time import time
import re



result_file=open('queries_op.txt','w',encoding='utf-8')

title_dict=defaultdict(str)
text_score_dict=defaultdict(str)
category_score_dict=defaultdict(str)
title_score_dict=defaultdict(str)
infobox_score_dict=defaultdict(str)
references_score_dict=defaultdict(str)
stopwords=defaultdict()


title_file=open('title_names.txt','r',encoding='utf-8')

body_scores_file=open('scored_body_index.txt','r',encoding='utf-8')
title_scores_file=open('scored_title_index.txt','r',encoding='utf-8')
infobox_scores_file=open('scored_infobox_index.txt','r',encoding='utf-8')
reference_scores_file=open('scored_reference_index.txt','r',encoding='utf-8')
category_scores_file=open('scored_category_index.txt','r',encoding='utf-8')



stopwords_file=open('stopwords2.txt','r')
stpwords=[line.rstrip() for line in stopwords_file]
for word in stpwords:
    stopwords[word]=True
stopwords_file.close()



for line in tqdm(title_file):
    try:
        ids,title=line.split('#')
        title_dict[int(ids)]=title.rstrip()
    except:
        pass


for line in tqdm(body_scores_file):
    word,scores=line.split('|')
    text_score_dict[word]=scores.rstrip()
    
for line in tqdm(reference_scores_file):
    word,scores=line.split('|')
    references_score_dict[word]=scores.rstrip()
    
for line in tqdm(category_scores_file):
    word,scores=line.split('|')
    category_score_dict[word]=scores.rstrip()
    
for line in tqdm(title_scores_file):
    word,scores=line.split('|')
    title_score_dict[word]=scores.rstrip()
    
for line in tqdm(infobox_scores_file):
    word,scores=line.split('|')
    infobox_score_dict[word]=scores.rstrip()
    
title_file.close()
body_scores_file.close()
reference_scores_file.close()
infobox_scores_file.close()
category_scores_file.close()
title_scores_file.close()


def is_phrase_query(query):
    ftq=['b:','c:','i:','r:','t:','l:']
    for q in ftq:
        if q in query:
            return False
    return True



def search_in_dictionary(word,dictionary_type):
    if dictionary_type=="body":
        return text_score_dict[word]
    
    if dictionary_type=="category":
        return category_score_dict[word]
    
    if dictionary_type=="infobox":
        return infobox_score_dict[word]
    
    if dictionary_type=="title":
        return title_score_dict[word]

    if dictionary_type=="references":
        return references_score_dict[word]



def get_one_word_scores(word,field_type):
    word=nltk_stemmer.stem(word)
    all_scores=search_in_dictionary(word,field_type)
    
    if all_scores != "":
        scores=all_scores.split(',')
        return scores
    else:
        return ""



def search_worst_case_query(query,num_results,sorted_ids,ordered_score_list):
    worst_case_id_score=defaultdict(float)
    print('got through worst case')
    new_ids=[ids for id_lst in sorted_ids for ids in id_lst ]
    
    for ids in new_ids:
        for dicts in ordered_score_list:
            worst_case_id_score[ids]+=dicts[ids]
    
    new_id_scores=sorted(worst_case_id_score.items(),key=operator.itemgetter(1),reverse=True)
    
    return new_id_scores




def get_relevant_titles(intersected_ids,query,sorted_ids,num_results,ordered_score_list):
    copy=query
    query=query.lower()
    #print('printing via rel fun '+query)
    words_in_query=set(query.split())
    
    relevant_ids=[]
    
    relevant_id_score=defaultdict(float)
    
    for id_list in sorted_ids:
        for rel_id in id_list:
            title_for_id=title_dict[rel_id].lower()
            words_in_title=set(title_for_id.split())
            #print(words_in_title)
            if len(words_in_title.intersection(words_in_query))>=len(words_in_query):
                relevant_ids.append(rel_id)
    

    for ids in relevant_ids:
        for dicts in ordered_score_list:
            relevant_id_score[ids]+=dicts[ids]
    

    relevant_id_score=sorted(relevant_id_score.items(),key=operator.itemgetter(1),reverse=True)

    return relevant_id_score




def search_in_index(query,query_type,num_results):
    query=query.lstrip()
    query=query.rstrip()
    query=query.lower()
    words=list(set(query.split()))
    
    for word in stopwords.keys():
        try:
            words.remove(word)
        except:
            pass
    
    
    ordered_score_list=[]
    sorted_score_list=[]

    for word in words:
   
        ordered_score=defaultdict(float)

        word_scores=get_one_word_scores(word,query_type)
       
        if word_scores!="":
            for item in word_scores:
                ordered_score[int(item.split(':')[0])]=float(item.split(':')[1])


            ordered_score_list.append(ordered_score)

            ordered_set=sorted(ordered_score.items(),key=operator.itemgetter(1),reverse=True)

            sorted_score_list.append(ordered_set)
        else:
            return ""


    sorted_ids=[]
    for sorted_scores in sorted_score_list:
        ids=[item[0] for item in sorted_scores]

        sorted_ids.append(ids)

    
    #print(sorted_ids)
    
    intersected_ids=set(sorted_ids[0]).intersection(*sorted_ids)
    
    if len(intersected_ids) < num_results:
        return get_relevant_titles(intersected_ids,query,sorted_ids,num_results,ordered_score_list)

    
    total_scores=defaultdict(float)
    for ids in intersected_ids:
        for dicts in ordered_score_list:
            total_scores[ids]+=dicts[ids]
    
    #print('printing  normally '+query)

    final_scores=sorted(total_scores.items(),key=operator.itemgetter(1),reverse=True)

    return final_scores




def search_phrase_query(query,num_results):
    
    titles=search_in_index(query,'title',num_results)
    bodys=search_in_index(query,'body',num_results)
    refs=search_in_index(query,'references',num_results)
    cats=search_in_index(query,'category',num_results)
    infos=search_in_index(query,'infobox',num_results)
    
 
    
    phrase_dict=defaultdict(float)
   
    
    for item in titles:
        phrase_dict[item[0]]+=(item[1])
        
    for item in bodys:
        phrase_dict[item[0]]+=(item[1])
    
    for item in refs:
        phrase_dict[item[0]]+=(item[1])
    
    for item in cats:
        phrase_dict[item[0]]+=(item[1])
    
    for item in infos:
        phrase_dict[item[0]]+=(item[1])
    
    
    
    top_ids=sorted(phrase_dict.items(),key=operator.itemgetter(1),reverse=True)
    #print(top_ids)
    top_results=[(ids[0],title_dict[ids[0]]) for ids in top_ids]

    return top_results

def search_field_query(query,num_results):
    query_list=re.split('(b:|i:|c:|r:|t:|l:)',query)
    #print(query_list)
    reference_q=None
    text_q=None
    category_q=None
    infobox_q=None
    title_q=None
    link_q=None

    reference_query=None
    text_query=None
    category_query=None
    infobox_query=None
    title_query=None
    link_query=None
    try:
        reference_q=query_list.index('r:')
    except:
        pass
    
    try:
        text_q=query_list.index('b:')
    except:
        pass
    
    try:
        category_q=query_list.index('c:')
    except:
        pass
        
    
    try:
        infobox_q=query_list.index('i:')
    except:
        pass
    
    
    try:
        title_q=query_list.index('t:')
    except:
        pass
    
    try:
        link_q=query_list.index('l:')
    except:
        pass
    
    rf=0
    bf=0
    tf=0
    cf=0
    if_=0
    lf=0
    

    if reference_q:
        rf=1
        reference_query=query_list[reference_q+1]
    
    if link_q:
        lf=1
        link_query=query_list[link_q+1]
    
    if text_q:
        bf=1
        text_query=query_list[text_q+1]

    if category_q:
        cf=1
        category_query=query_list[category_q+1]

    if infobox_q:
        if_=True
        infobox_query=query_list[infobox_q+1]

    if title_q:
        tf=1
        title_query=query_list[title_q+1]

    
    ref_res=[]
    info_res=[]
    title_res=[]
    cat_res=[]
    link_res=[]
    text_res=[]
    
    ref_title=[]
    info_title=[]
    title_title=[]
    cat_title=[]
    link_title=[]

    body_title=[]
    
    intersected_ids=[]

    field_query_start_time=time()
    
    if(reference_q):
        #print('r ',reference_query.rstrip())
        ref_res=search_in_index(reference_query,'references',num_results)
        ref_title=[(ids[0],title_dict[ids[0]]) for ids in ref_res]
        intersected_ids.append([ids[0] for ids in ref_res])

    if(link_q):
        #print('r ',reference_query.rstrip())
        link_res=search_in_index(link_query,'references',num_results)
        link_title=[(ids[0],title_dict[ids[0]]) for ids in link_res]
        intersected_ids.append([ids[0] for ids in link_res])
    
    if(infobox_q):
        #print('i ',infobox_query.rstrip())
        info_res=search_in_index(infobox_query,'infobox',num_results)
        info_title=[(ids[0],title_dict[ids[0]]) for ids in info_res]
        intersected_ids.append([ids[0] for ids in info_res])    
        
    if(text_q):
        #print('t ',text_query.rstrip())
        text_res=search_in_index(text_query,'body',num_results)
        body_title=[(ids[0],title_dict[ids[0]]) for ids in text_res]
        intersected_ids.append([ids[0] for ids in text_res])

    if(title_q):
        #print('T ',title_query.rstrip())
        title_res=search_in_index(title_query,'title',num_results)
        
        title_title=[(ids[0],title_dict[ids[0]]) for ids in title_res]
        intersected_ids.append([ids[0] for ids in title_res])
    
    if(category_q):
        #print('c ',category_query.rstrip())
        cat_res=search_in_index(category_query,'category',num_results)
        cat_title=[(ids[0],title_dict[ids[0]]) for ids in cat_res]
        intersected_ids.append([ids[0] for ids in cat_res])
        
    
    
    final_results=set(intersected_ids[0]).intersection(*intersected_ids)

    extras=[]

    results=[(item,title_dict[item]) for item in final_results]

    if len(results) >= num_results:
          field_query_end_time=time()

          total_time=field_query_end_time - field_query_start_time
          average_time_per_query=(total_time)/(if_+lf+tf+cf+bf+rf)
          
          return results,total_time,average_time_per_query

    else:
        if tf > 0:
            extras.append(search_phrase_query(title_query,num_results)[:num_results])

        if if_ > 0:
            extras.append(search_phrase_query(infobox_query,num_results)[:num_results])
        
        if cf > 0:
            extras.append(search_phrase_query(category_query,num_results)[:num_results])
        
        if lf > 0:
            extras.append(search_phrase_query(link_query,num_results)[:num_results])
        
        if rf > 0:
            extras.append(search_phrase_query(reference_query,num_results)[:num_results])
        
        if bf > 0:
            extras.append(search_phrase_query(text_query,num_results)[:num_results])
                

        union_results=list(set(extras[0]).union(*extras))
        
       

        results=results+union_results

        field_query_end_time=time()

        total_time=field_query_end_time - field_query_start_time
        average_time_per_query=(total_time)/(if_+lf+tf+cf+bf+rf)

    return results,total_time,average_time_per_query


def execute_query(query,num_results):
    if is_phrase_query(query):
        start=time()
        phrase_result=search_phrase_query(query,num_results)
        end=time()

        for i in range(num_results):
            try:
                result_file.write(f'{phrase_result[i][0]},{phrase_result[i][1]}\n')
            except:
                pass
        result_file.write(f'{end-start},{end-start}\n')
        result_file.write('\n')


        #print(f'Phrase Query Results \n {phrase_result[:num_results]}')
    
    else:
        #title_title,body_title,cat_title,info_title,ref_title,link_title=search_field_query(query,num_results)
        
        top_field_scores,total_time,average_time_per_query=search_field_query(query,num_results)
        


        for i in range(num_results):
            try:
                result_file.write(f'{top_field_scores[i][0]},{top_field_scores[i][1]}\n')
            except:
                pass

        result_file.write(f'{total_time},{average_time_per_query}\n')
        result_file.write('\n')
        # print(top_field_scores[:num_results])
        # print(total_time,average_time_per_query)



query_file=sys.argv[1]


query_file=open(query_file,'r',encoding='utf-8')
for queries in query_file:
    full_query=queries.split(',')
    num_results=int(full_query[0])
    query=' '.join(full_query[1:])
    #num_results,query=queries.split(',')
    print(num_results,query.strip())
    execute_query(query.rstrip(),int(num_results))
    print('')

query_file.close()


result_file.close()