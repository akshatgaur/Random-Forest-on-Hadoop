The Complete Process for Random Forest is divided into three parts.
1. Preprocessing of Data
	In preprocessing I have created train file for each mapper function. These train files have 2/3rd of train data and 6 features.
	These train files should be pushed to hdfs directory. Along with these train file we also need to push test file to hdfs for mapper.
 
2. Random Forest over MapReduce
	In this part, multiple trees are created over each mapper node. 
	These mappers also have the test data and so perform the prediction using the tree created.
	In reducer, the predictions given by all the mappers are used to find the best predicted label.

3. Accuracy and other metric calculation
	Finally calculate the accuracy by calculating the misclassification and also calculate other metrics.
	

1. Preprocessing of Data:
	
	a. Execute python script CreateInputFiles.py
		python CreateInputFiles.py <datasource_file> <Num of trees> <hdfs directory path>
		python CreateInputFiles.py dataset-har-PUC-Rio-ugulino.csv 3 hdfs://aspen.local/user/agaur/RandomForest/
	b. After this push all the trainFile_<I>.csv to hdfs to the path provided above.
	   Push the test file to the same hdfs directory.

2. Random Forest over MapReduce
	
	a. Execute RandomForest.py script as follows
		python RandomForest.py <input file for mapper> -r hadoop > result
	This input file for mapper is created by previous script so it will be below value only.
		python RandomForest.py mapper_input_file.txt -r hadoop > result

3. Accuracy and other metrics
	a. Execute accuracy.py script as follows
		python accuracy.py <predictions> <true_labels>
		python accuracy.py result labels