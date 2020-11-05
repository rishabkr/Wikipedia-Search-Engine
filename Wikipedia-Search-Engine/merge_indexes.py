import os
from heapq import *
from collections import defaultdict
import math
from tqdm import tqdm
import operator
from time import time
import sys


class IndexMerger:
    def __init__(self,top_k,doc_count):
        
        self.body_directory='indexes/body/'
        self.category_directory='indexes/category/'
        self.references_directory='indexes/references/'
        self.infobox_directory='indexes/infobox/'
        self.title_directory='indexes/titles/'


        self.body_files=os.listdir(self.body_directory)
        self.body_index='final_text_index.txt'
        

        self.title_files=os.listdir(self.title_directory)
        self.title_index='final_title_index.txt'

        self.category_files=os.listdir(self.category_directory)
        self.category_index='final_category_index.txt'

        self.reference_files=os.listdir(self.references_directory)
        self.reference_index='final_reference_index.txt'


        self.infobox_files=os.listdir(self.infobox_directory)
        self.infobox_index='final_infobox_index.txt'

        self.top_k=top_k
        self.num_pages=doc_count


    def get_tf_idf(self,word,postings,index_writer,k):
         
        word_idf=defaultdict(float)

        docs=postings.split(',')
       
        idf=math.log10(self.num_pages/len(docs))
        for doc in docs:
                doc_num,count=doc.split(':')
                tf=1+math.log10(int(count))
                word_idf[str(doc_num)]=round(idf*tf,2)

        
        word_idf=sorted(word_idf.items(),key=operator.itemgetter(1),reverse=True)
 
        final_doc=[str(item[0])+':'+str(item[1]) for item in word_idf]


        final_doc=final_doc[:k+1]
        
        tf_idf_index=','.join(final_doc)+'\n'
        
        tf_idf_index=word+'|'+tf_idf_index
        
        index_writer.write(tf_idf_index)

 
    def merge_index_files(self,heap,file_pointers,index_type,index_files,k):
        
        final_index=open(index_type,'w',encoding='utf-8')
        num_files=len(index_files)
       
        line_number=0
        file_count=0
        heapify(heap)

        
        
        try:
            while(file_count<=num_files):

                posting_list,file_number=heappop(heap)
                word,postings=posting_list.split('|')
            
                fp=file_pointers[int(file_number)]
                next_line=fp.readline().rstrip()
                
            
                if next_line:
                    heappush(heap,(next_line,file_number))
                
                else:
                    fp.close()
                    file_count=file_count+1

                
                if file_count==num_files:
                    break

            
                while(1):
                    posts,file_number=heappop(heap)
                
                    curr_word,curr_postings=posts.split('|')

                    if curr_word==word:
                        postings=postings+","+curr_postings
                        fp=file_pointers[int(file_number)]
                        next_line=fp.readline().rstrip()
                        
                        if next_line:
                            heappush(heap,(next_line,file_number))

                        else:
                            fp.close()
                            file_count=file_count+1
                    else:
                        heappush(heap,(posts,file_number))
                        break;

                self.get_tf_idf(word,postings,final_index,k)
        except:
            pass
        final_index.close()


    
    def create_scored_index(self,index_file,scored_index_file_name,top_k):
        big_index=open(index_file,'r',encoding='utf-8')
        large_dict=defaultdict(str)

        for lines in big_index:
            word,postings=lines.split('|')
            if(large_dict.get(word) is not None):
                    postings=','+postings
            large_dict[word]+=postings.rstrip()
        
        big_index.close()

        new_dict=defaultdict(list)
        
        for word in tqdm(large_dict.keys()):
            posts=large_dict[word].split(',')
    
            postings=[(str(docs.split(':')[0]),float(docs.split(':')[1])) for docs in posts]
            top=sorted(postings,key=operator.itemgetter(1),reverse=True)
            
            # if(len(top)>top_k):
            #     new_dict[word]=top[:top_k+1]
            # else:
            #     new_dict[word]=top

            new_dict[word]=top[:top_k+1]

        scored_index=open(scored_index_file_name,'w',encoding='utf-8')

        for word in tqdm(sorted(new_dict.keys())):
            curr=""
            for items in new_dict[word]:
                curr+=items[0]+':'+str(items[1])+','
            
            scored_index.write(word+'|'+curr.rstrip(',')+'\n')

        scored_index.close()


    def merge_body_files(self):
        self.body_file_pointers={}
        
        i=1
        heap=[]

        for fname in self.body_files:
            fp=open(self.body_directory+fname,'r',encoding='utf-8')
            self.body_file_pointers[i]=fp
            first_line=fp.readline().rstrip()
            heap.append((first_line,i))
            i+=1

        start=time()
        self.merge_index_files(heap,self.body_file_pointers,self.body_index,self.body_files,self.top_k)
        end=time()

        print(f'Time taken to construct merged body files {end-start}')
        

        start2=time()
       
        self.create_scored_index(self.body_index,"scored_body_index.txt",self.top_k)

        end2=time()
        print(f'time taken to create scored body index {end2-start2}')
        heap=[]
        
        self.clean_up("body")

    
    def merge_category_files(self):
        self.category_file_pointers={}
        
        i=1
        heap=[]

        for fname in self.category_files:
            fp=open(self.category_directory+fname,'r',encoding='utf-8')
            self.category_file_pointers[i]=fp
            first_line=fp.readline().rstrip()
            heap.append((first_line,i))
            i+=1

        start=time()
        self.merge_index_files(heap,self.category_file_pointers,self.category_index,self.category_files,1000000)

        end=time()

        print(f'Time taken to construct merged category_files files {end-start}')
        

        start2=time()
       
        self.create_scored_index(self.category_index,"scored_category_index.txt",1000000)

        end2=time()
        print(f'time taken to create scored category index {end2-start2}')
        heap=[]
        
        self.clean_up("category")
    
    
    def merge_infobox_files(self):
        self.infobox_file_pointers={}
        
        i=1
        heap=[]

        for fname in self.infobox_files:
            fp=open(self.infobox_directory+fname,'r',encoding='utf-8')
            self.infobox_file_pointers[i]=fp
            first_line=fp.readline().rstrip()
            heap.append((first_line,i))
            i+=1

        start=time()
        self.merge_index_files(heap,self.infobox_file_pointers,self.infobox_index,self.infobox_files,1000000)

        end=time()

        print(f'Time taken to construct merged infobox files {end-start}')
        

        start2=time()
       
        self.create_scored_index(self.infobox_index,"scored_infobox_index.txt",1000000)

        end2=time()
        print(f'time taken to create scored infobox index {end2-start2}')
        heap=[]
        
        self.clean_up("infobox")

    
    def merge_title_files(self):
        self.title_file_pointers={}
        
        i=1
        heap=[]

        for fname in self.title_files:
            fp=open(self.title_directory+fname,'r',encoding='utf-8')
            self.title_file_pointers[i]=fp
            first_line=fp.readline().rstrip()
            heap.append((first_line,i))
            i+=1

        start=time()
        self.merge_index_files(heap,self.title_file_pointers,self.title_index,self.title_files,1000000)

        end=time()

        print(f'Time taken to construct merged title files {end-start}')
        

        start2=time()
       
        self.create_scored_index(self.title_index,"scored_title_index.txt",1000000)

        end2=time()
        print(f'time taken to create scored title index {end2-start2}')
        heap=[]
        
        self.clean_up("title")



    def merge_reference_files(self):
        self.reference_file_pointers={}
        
        i=1
        heap=[]

        for fname in self.reference_files:
            fp=open(self.references_directory+fname,'r',encoding='utf-8')
            self.reference_file_pointers[i]=fp
            first_line=fp.readline().rstrip()
            heap.append((first_line,i))
            i+=1

        start=time()
        self.merge_index_files(heap,self.reference_file_pointers,self.reference_index,self.reference_files,2500)

        end=time()

        print(f'Time taken to construct merged reference files {end-start}')
        

        start2=time()
       
        self.create_scored_index(self.reference_index,"scored_reference_index.txt",2500)

        end2=time()
        print(f'time taken to create scored reference index {end2-start2}')
        heap=[]
        self.clean_up("reference")




    def clean_up(self,clean_up_type):
        if clean_up_type=="body":

            os.remove(self.body_index)

            for i in self.body_file_pointers:
                self.body_file_pointers[i].close()

            #for fname in self.body_files:
                #os.remove(self.body_directory+fname)

        if clean_up_type=="reference":

            os.remove(self.reference_index)

            for i in self.reference_file_pointers:
                self.reference_file_pointers[i].close()

            #for fname in self.reference_files:
                #os.remove(self.references_directory+fname)

        
        if clean_up_type=="category":

            os.remove(self.category_index)

            for i in self.category_file_pointers:
                self.category_file_pointers[i].close()

            #for fname in self.category_files:
                #os.remove(self.category_directory+fname)


        if clean_up_type=="infobox":

            os.remove(self.infobox_index)

            for i in self.infobox_file_pointers:
                self.infobox_file_pointers[i].close()

            #for fname in self.infobox_files:
                #os.remove(self.infobox_directory+fname)


        if clean_up_type=="title":

            os.remove(self.title_index)

            for i in self.title_file_pointers:
                self.title_file_pointers[i].close()

            #for fname in self.title_files:
                #os.remove(self.title_directory+fname)



    def merge_files(self):
        self.merge_title_files()
        self.merge_infobox_files()
        self.merge_category_files()
        self.merge_body_files()
        self.merge_reference_files()


if __name__=='__main__':
    start=time()
    doc_count=0
    tf=open('title_names.txt','r',encoding='utf-8')
    for line in tf:
        doc_count+=1
    tf.close()
    print(doc_count)
    
    merger=IndexMerger(520,doc_count)
    
    merger.merge_files()
    
    end=time()

    print(f'Total time taken to merge is {end-start} seconds!! ')