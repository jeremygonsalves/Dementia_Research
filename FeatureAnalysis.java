import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class FeatureAnalysis {
    public static void main(String[] args) {
        // Step 1: Initialize a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("FeatureAnalysis")
                .getOrCreate();

        // Step 2: Load your data into a DataFrame
        // Replace "your_data.csv" with the path to your CSV file
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("your_data.csv");

        // Step 3: Define the feature columns (excluding 'Participant #' and 'PlayAttention Percentage')
        String[] featureColumns = data.columns();
        String[] inputCols = java.util.Arrays.stream(featureColumns)
                .filter(col -> !col.equals("Participant #") && !col.equals("PlayAttention Percentage"))
                .toArray(String[]::new);

        // Step 4: Assemble features into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(data);

        // Step 5: Feature Selection using Chi-Squared Selector
        // Using 'PlayAttention Percentage' as the target label
        ChiSqSelector selector = new ChiSqSelector()
                .setNumTopFeatures(10)
                .setFeaturesCol("features")
                .setLabelCol("PlayAttention Percentage")
                .setOutputCol("selectedFeatures");

        Dataset<Row> selectedData = selector.fit(assembledData).transform(assembledData);

        // Step 6: Apply PCA for Dimensionality Reduction
        PCA pca = new PCA()
                .setK(2)
                .setInputCol("features")
                .setOutputCol("pcaFeatures");

        Dataset<Row> pcaResult = pca.fit(selectedData).transform(selectedData).select("pcaFeatures");

        // Step 7: Show the PCA results
        pcaResult.show(false);

        // Optional: Save the PCA results
        pcaResult.write().csv("pca_output.csv");

        // Stop the Spark session
        spark.stop();
    }
}
