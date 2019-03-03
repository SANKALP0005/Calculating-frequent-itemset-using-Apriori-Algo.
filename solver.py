def solve(xyz,K,F): #dataset,K,F
    
	words = [""]
	f = open("vocab."+xyz+".txt","r")
	for w in (f.read().split()):#this loop will add all the words in vocab to list 
	    words.append(str(w))
	f.close()
	f2 = open("docword."+xyz+".txt","r")#for reading the no. of docs,words,entries and wid that appears in particular docid
	a = f2.readlines()
	f2.close()

	ndocs = int(a[0])
	nwords = int(a[1])
	nentries = int(a[2])
	F=int(F*nentries)
	rows = [] # list of itemsets(set of words) for each document(entry)
	for i in range(1,ndocs+1):
	    rows.append([])
	    
	for i in range(3,nentries+3):#mapping words with their respective id
	    row = a[i].split()
	    docid = int(row[0])
	    wordid = int(row[1])
	    rows[docid-1].append(words[wordid])

	import pandas as pd
	from mlxtend.frequent_patterns import apriori
	from mlxtend.preprocessing import TransactionEncoder


	te = TransactionEncoder()
	te_ary = te.fit(rows).transform(rows)
	df = pd.DataFrame(te_ary, columns=te.columns_)

	import time
	stime = time.time()
	sup = (F/nentries)

	frequent_itemsets = apriori(df, min_support= sup, use_colnames=True)
	frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)) == K] #As this pakage gives output for all possible frequent 
	#itemset with given threshold, so apply filter to get only required k
	#print( frequent_itemsets )

	path = 'D:\\COLLEGE\\Data_set\\Output\\' + xyz + '\\K=' + str(K) + '_' + 'F=' + str(F) + '.txt' #give path to save ouput to this file
	fo = open(path,"w")
	for index,row in frequent_itemsets.iterrows():
	    for j in row['itemsets']:
	        fo.write(str(j) + ',')
	    fo.write('\n')
	fo.close()
	etime = time.time() - stime
	path2 = 'D:\\COLLEGE\\Data_set\\ExecutionTimes\\' + xyz + '.txt'; #give path where to save execution times 
	ft = open(path2,"a")
	ft.write("K=%s , F=%s --> %s sec\n" % (K,F,etime))
	ft.close()
