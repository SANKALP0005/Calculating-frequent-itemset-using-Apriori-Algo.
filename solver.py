def solve(xyz,K,F): #dataset,K,F
    
	words = [""]
	f = open("vocab."+xyz+".txt","r")
	for w in (f.read().split()):
	    words.append(str(w))
	f.close()
	f2 = open("docword."+xyz+".txt","r")
	a = f2.readlines()
	f2.close()

	ndocs = int(a[0])
	nwords = int(a[1])
	nentries = int(a[2])
	F=int(F*nentries)
	rows = [] # list of itemsets(set of words) for each document(entry)
	for i in range(1,ndocs+1):
	    rows.append([])
	    
	for i in range(3,nentries+3):
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
	frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)) == K] #filter
	#print( frequent_itemsets )

	path = 'D:\\COLLEGE\\Data_set\\Output\\' + xyz + '\\K=' + str(K) + '_' + 'F=' + str(F) + '.txt' #save ouput to this file
	fo = open(path,"w")
	for index,row in frequent_itemsets.iterrows():
	    for j in row['itemsets']:
	        fo.write(str(j) + ',')
	    fo.write('\n')
	fo.close()
	etime = time.time() - stime
	path2 = 'D:\\COLLEGE\\Data_set\\ExecutionTimes\\' + xyz + '.txt'; #save execution time here
	ft = open(path2,"a")
	ft.write("K=%s , F=%s --> %s sec\n" % (K,F,etime))
	ft.close()