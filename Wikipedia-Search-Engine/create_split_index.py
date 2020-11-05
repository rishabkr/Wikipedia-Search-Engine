import re
from collections import defaultdict
import xml.etree.ElementTree as etree
import nltk
nltk_stemmer=nltk.stem.SnowballStemmer('english')
from tqdm import tqdm
import sys
import os
from time import time

# INDEX_FILE='inverted_index.txt'
# WIKI_XML_FILE_NAME='enwiki-latest-pages-articles2.xml'

# sample_xml='sample.xml'
# title_file='title_index.txt'
class Wiki_Indexer:
    def __init__(self):
        self.title_index=defaultdict(list)
        self.text_index=defaultdict(list)
        self.category_index=defaultdict(list)
        #title_tag_file=open(title_file,'w',encoding='utf-8')
        
        self.title_positions=[]
        self.stopwords=defaultdict()
        self.infobox_index=defaultdict(list)
        self.reference_index=defaultdict(list)
        self.stemmed_words={}
        self.css_pattern=re.compile(r'{\|(.*?)\|}',re.DOTALL)
        self.file_pattern = re.compile(r'\[\[file:(.*?)\]\]',re.DOTALL)
        self.url_pattern=re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.DOTALL)
        
        self.STOPWORDS_FILE='stopwords2.txt'

        self.title_index_file=open('title_names.txt','w',encoding='utf-8')

        self.total_tokens_in_dump=0
        self.total_tokens_in_index=0
        
        self.page_count=0
        
        self.dump_size=0

   
        self.num_file=0

        self.num_files=12
    
    def load_stopwords(self):
            stopwords_file=open(self.STOPWORDS_FILE,'r')
            stpwords=[line.rstrip() for line in stopwords_file]
            for word in stpwords:
                self.stopwords[word]=True
            stopwords_file.close()
    
    def add_to_stem_dict(self,word):
       
        self.stemmed_words[word]=nltk_stemmer.stem(word)
        return self.stemmed_words[word]
        
    def parse_category_box(self,text,category_count):
            categories=re.findall("\[\[Category:(.*?)\]\]",text)
      
            for category in categories:
                text_pattern=re.compile(r'[a-zA-Z]+|[0-9]{1,4}') 
                category_words=re.findall(text_pattern,category)
                category_words=list(map(lambda x:x.lower(),category_words))
                
                category_words=[x for x in category_words if (x!="" and (not x in self.stopwords and len(x)) and len(x)>2)]
                #self.total_tokens_in_dump+=len(category_words)
                category_words=[self.stemmed_words[word] if word in self.stemmed_words else self.add_to_stem_dict(word) for word in category_words]
                # print(category_words)
                for word in category_words:
                     category_count[word]+=1
            
            return category_count
    

        
    def parse_infobox_text(self,text,infobox_page_count):
     
                infobox_text=[]
                flag=False
                for line in text.split('\n'):
                    if line.startswith("{{Infobox"):
                        flag=True
                        infobox_text.append(line)
                    elif flag and line!="}}":
                        infobox_text.append(line)
                    elif flag:
                        break
                
                for info_text in infobox_text:
                    text_pattern=re.compile(r'[a-zA-Z]+|[0-9]{1,4}') 
                    info_words=re.findall(text_pattern,info_text)
                    
                    info_words=list(map(lambda x:x.lower(),info_words))
                    
                    info_words=[x for x in info_words if (x!="" and (not x in self.stopwords)and len(x)>2)]
                    #self.total_tokens_in_dump+=len(info_words)

                    info_words=[self.stemmed_words[word] if word in self.stemmed_words else self.add_to_stem_dict(word) for word in info_words]
                    for word in info_words:
                        infobox_page_count[word]+=1
                return infobox_page_count
    
                
    def parse_main_text(self,text,main_text_count,category_count,infobox_page_count,reference_count):
            #text=css_pattern.sub('',text)
            body=text.lower()
           
            text_pattern=re.compile(r'[a-zA-Z]+|[0-9]{1,4}') 
            body=re.findall(text_pattern,body)
            body_words=[x for x in body if (x!="" and (not x in self.stopwords)and len(x)>2)]
            
            self.total_tokens_in_dump+=len(body_words)

            body_words=[self.stemmed_words[word] if word in self.stemmed_words else self.add_to_stem_dict(word) for word in body_words]
            for word in body_words:
                main_text_count[word]+=1
            
            for word in main_text_count.keys():
                 
                if(category_count.get(word) is not None):
                    main_text_count[word]-=category_count.get(word)

                if(infobox_page_count.get(word) is not None):
                    main_text_count[word]-=infobox_page_count.get(word)


                if(reference_count.get(word)is not None):
                    main_text_count[word]-=reference_count.get(word)


            return main_text_count
                
    def parse_title(self,title,title_count):

            ttle=title
            ttle+='\n'
            
            body=title.lower()
            text_pattern=re.compile(r'[a-zA-Z]+|[0-9]{1,4}') 
            body=re.findall(text_pattern,body)
            body_words=[x for x in body if (x!="" and len(x)>2 and (not x in self.stopwords))]
            
            self.total_tokens_in_dump+=len(body_words)
            body_words=[self.stemmed_words[word] if word in self.stemmed_words else self.add_to_stem_dict(word) for word in body_words]
            
            for word in body_words:
                title_count[word]+=1
            return title_count

    def parse_for_references(self,text,reference_count):
  

        text_pattern=re.compile("[a-zA-Z\d]+")
        st1=re.sub('&lt;ref&gt;','<ref>',text)
        st2=re.sub('&lt;/ref&gt;','</ref>',st1)
        lists=re.findall('\<ref\>([^\<]+)\<\/ref\>',st2)
        
        words=[re.findall(text_pattern,element) for element in lists]
        nonempty_words=[[word for word in list1 if(word!="" and len(word)>2 and (not word in self.stopwords))] for list1 in words]
        words=[word.lower() for lists in nonempty_words for word in lists]
        #self.total_tokens_in_dump+=len(words)

        words=[self.stemmed_words[word] if word in self.stemmed_words else self.add_to_stem_dict(word) for word in words]
        
        for word in words:
            reference_count[word]+=1
        return reference_count
    
    def write_title_index(self,id_,count,title_count):
         
            for word in title_count:
                word_count=str(title_count[word])
                page_number=str(count)
                word_and_page_count=id_+':'+word_count
                self.title_index[word].append(word_and_page_count)
        
    def write_text_index(self,id_,count,main_text_count):
           for word in main_text_count:
                if main_text_count[word]>0:
                    word_count=str(main_text_count[word])
                    page_number=str(count)
                    word_and_page_count=id_+':'+word_count
                    self.text_index[word].append(word_and_page_count)
        
        
    def write_category_index(self,id_,count,category_count):
            for word in category_count:
                if category_count[word]>0:
                    word_count=str(category_count[word])
                    page_number=str(count)
                    word_and_page_count=id_+':'+word_count
                    self.category_index[word].append(word_and_page_count)
        

    def write_reference_index(self,id_,count,reference_count):
          
            for word in reference_count:
            	if reference_count[word]>0:
	                word_count=str(reference_count[word])
	                page_number=str(count)
	                word_and_page_count=id_+':'+word_count
	                self.reference_index[word].append(word_and_page_count)
       

                
    def write_infobox_index(self,id_,count,infobox_page_count):
            for word in infobox_page_count:
                if infobox_page_count[word]!=0:
                    word_count=str(infobox_page_count[word])
                    page_number=str(count)
                    word_page_count=id_+':'+word_count

                    self.infobox_index[word].append(word_page_count)
                else:
                    continue


    def get_params(self):
            param=sys.argv
            self.collectionFile='xml_dump.xml'#param[1]
            self.indexFile='index_file.txt'#param[2]
            self.stat_file='index_stat.txt'#param[3]
            
    def parse_xml_file(self):
            from time import time

            
            #self.get_params()
            self.load_stopwords()
            count=0
            revsn_tag=False
            #WIKI_XML_FILE_NAME
            
            self.num_file=1
            dump_names=os.listdir(sys.argv[1])
            file_name=sys.argv[1]+'/'
            print(dump_names)
            for i in range(len(dump_names)):
                #self.collectionFile=file_name+'_'+str(i)+'.xml'
                self.collectionFile=dump_names[i]
          
                print(f'Processing dump number: {i}')
                start=time()
                with open(file_name+self.collectionFile,encoding='utf-8') as xmlfile:
                    context=etree.iterparse(xmlfile,events=('start','end'))
                    context=iter(context)
                    
                    for event, element in tqdm(context):
                        tag_name=re.sub(r'{.*}','',element.tag)
                        if event=='start':
                            if tag_name=='page':
                                category_count=defaultdict(int)
                                title_count=defaultdict(int)
                                main_text_count=defaultdict(int)
                                infobox_page_count=defaultdict(int)
                                reference_count=defaultdict(int)
                                revsn_tag=False
                                self.page_count+=1
                            elif tag_name=='revision':
                                revsn_tag=True

                        else:
                            if tag_name=='title' :
                                title_text=element.text
                                title_count=self.parse_title(title_text,title_count)

                            elif tag_name=='id' and not revsn_tag:
                                ids=int(element.text)

                            elif tag_name=='text':
                                try:
                                    current_text=element.text
                                    current_text=self.css_pattern.sub('',current_text)
                                    current_text=self.file_pattern.sub('',current_text)
                                    current_text=self.url_pattern.sub('',current_text)

                                    category_count=self.parse_category_box(current_text,category_count)
                            
                                    infobox_page_count=self.parse_infobox_text(current_text,infobox_page_count)
                                    reference_count=self.parse_for_references(current_text,reference_count)
                                    main_text_count=self.parse_main_text(current_text,main_text_count,category_count,infobox_page_count,reference_count)
                            
                                except:
                                    pass

                            elif tag_name=='page':
                                self.write_title_index(str(ids),count,title_count)
                                self.write_category_index(str(ids),count,category_count)
                                self.write_text_index(str(ids),count,main_text_count)
                                self.write_infobox_index(str(ids),count,infobox_page_count)
                                self.write_reference_index(str(ids),count,reference_count)
                                self.title_index_file.write(str(ids)+'#'+title_text+'\n')
                         

                                if(self.page_count%30000==0):
                                    self.write_index_files(self.num_file)
                                    self.num_file+=1
                                    self.text_index.clear()
                                    self.title_index.clear()
                                    self.infobox_index.clear()
                                    self.category_index.clear()
                                    self.reference_index.clear()
                                
                                if(self.page_count%50000==0):
                                    self.stemmed_words.clear()

                            element.clear()

                   

                    end=time()
                    print(f'Parsing successful!! in {end-start} seconds ')
                    
                    self.write_index_files(self.num_file)
                    self.num_file+=1

                    self.text_index.clear()
                    self.title_index.clear()
                    self.infobox_index.clear()
                    self.category_index.clear()
                    self.reference_index.clear()

                    xmlfile.close()
            self.title_index_file.close()
                    
                    #title_tag_file.close()


    def write_index_files(self,idx):

            category_set=sorted(set(self.category_index.keys()))
            title_set=sorted(set(self.title_index.keys()))
            text_set=sorted(set(self.text_index.keys()))
            reference_set=sorted(set(self.reference_index.keys()))
            infobox_set=sorted(set(self.infobox_index.keys()))


            for word in self.stopwords.keys():
                    try:
                        category_set.remove(word)
                        title_set.remove(word)
                        text_set.remove(word)
                        reference_set.remove(word)
                        infobox_set.remove(word)
                    except:
                        pass
                    
                    
                    
            body_directory='indexes/body/'
            category_directory='indexes/category/'
            references_directory='indexes/references/'
            infobox_directory='indexes/infobox/'
            title_directory='indexes/titles/'

            category_index_file=category_directory+'category_index_'+str(idx)+'.txt'
            infobox_index_file=infobox_directory+'infobox_index_'+str(idx)+'.txt'
            reference_index_file=references_directory+'reference_index_'+str(idx)+'.txt'
            body_index_file=body_directory+'body_index_'+str(idx)+'.txt'
            title_index_file=title_directory+'title_index_'+str(idx)+'.txt'

            cif=open(category_index_file,'w',encoding='utf-8')
            iif=open(infobox_index_file,'w',encoding='utf-8')
            rif=open(reference_index_file,'w',encoding='utf-8')
            bif=open(body_index_file,'w',encoding='utf-8')
            tif=open(title_index_file,'w',encoding='utf-8')

            self.total_tokens_in_index=self.total_tokens_in_index+len(category_set)+len(title_set)+len(text_set)+len(reference_set)+len(infobox_set)
              
            for i in category_set:
                            cat_posting=self.category_index[i]
                            cat_posting=','.join(cat_posting)
                            if cat_posting:
                                cif.write(f'{i}|{cat_posting}\n')

            for i in infobox_set:
                            info_posting=self.infobox_index[i]
                            info_posting=','.join(info_posting)
                            if info_posting:
                                iif.write(f'{i}|{info_posting}\n')
            
            for i in reference_set:
                            reference_posting=self.reference_index[i]
                            reference_posting=','.join(reference_posting)
                            if reference_posting:
                                rif.write(f'{i}|{reference_posting}\n')
            
            for i in  text_set:
                            text_posting=self.text_index[i]
                            text_posting=','.join(text_posting)
                            if text_posting:
                                bif.write(f'{i}|{text_posting}\n')                    

            for i in title_set:
                            title_posting=self.title_index[i]
                            title_posting=','.join(title_posting)
                            if title_posting:
                                tif.write(f'{i}|{title_posting}\n')


            cif.close()
            bif.close()
            tif.close()
            iif.close()
            rif.close()




            # stat_file=open(f'index_stat_{idx}.txt','w')
            # stat_file.write(str(self.total_tokens_in_dump)+'\n')
            # stat_file.write(str(self.total_tokens_in_index)+'\n')
            # stat_file.close()



def get_size(start_path = 'indexes'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


if __name__=='__main__':
    indexer=Wiki_Indexer()

    os.makedirs('indexes/body')
    os.makedirs('indexes/titles')
    os.makedirs('indexes/references')
    os.makedirs('indexes/category')
    os.makedirs('indexes/infobox')
    

    start=time()
    indexer.parse_xml_file()
    end=time()
    

    index_size=get_size()/2**30

   
    stat_file=open('stats.txt','w',encoding='utf-8')
    stat_file.write(f'{index_size} GB\n')
    stat_file.write(f'{indexer.num_file-1}\n')
    stat_file.write(f'{indexer.total_tokens_in_index}\n')
   
    stat_file.close()

    


    print(f'Total time taken is {end-start} Seconds')