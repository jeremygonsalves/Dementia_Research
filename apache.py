from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, ChiSqSelector, PCA
from pyspark.ml import Pipeline
from pyspark.sql.functions import col


import pandas as pd
from sklearn.preprocessing import OneHotEncoder




# Initialize a Spark session
spark = SparkSession.builder \
    .appName("FeatureAnalysis") \
    .getOrCreate()

# Load your data
data = spark.read.option("header", "true").option("inferSchema", "true").csv("cleaned_data7.csv")

# Specify the target column
target_column = 'PlayAttention Percentage'

# Identify the columns to be used as features, excluding 'Participant #' and the target column
columns_to_exclude = ['Participant #', target_column]
feature_columns = [col for col in data.columns if col not in columns_to_exclude]

# Identify categorical columns (string type)
categorical_columns = [col for col in feature_columns if str(data.schema[col].dataType) == "StringType"]

# StringIndexer + OneHotEncoder for categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_ohe") for col in categorical_columns]

# Assemble the feature vector
assembler = VectorAssembler(
    inputCols=[col+"_ohe" for col in categorical_columns] + 
              [col for col in feature_columns if col not in categorical_columns],
    outputCol="features"
)

# Chi-Square Selector (optional, based on your needs)
selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol=target_column)

# PCA for dimensionality reduction
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")

# Create a pipeline to streamline the process
pipeline = Pipeline(stages=indexers + encoders + [assembler, selector, pca])

# Fit the pipeline to the data
model = pipeline.fit(data)

# Transform the data
result = model.transform(data)

# Show the PCA result
result.select("pcaFeatures").show(truncate=False)

# Stop the Spark session
spark.stop()
