

def convert_pandas_to_spark_with_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler


    inc=20000
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):
        if isinstance (a_dataframe,  pd.DataFrame):
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
        else:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].toarray())

        a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names)  )
        
        vecAssembler = VectorAssembler(outputCol="features")
        vecAssembler.setInputCols(c_names)
        a_spark_vector = vecAssembler.transform(a_df)
        
        if bool == True:
            spark_df = a_spark_vector
            bool = False
        else:
            spark_df = spark_df.union(a_spark_vector)
    return spark_df


def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names