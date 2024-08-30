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
        SparkSession spark = SparkSession.builder()
                .appName("FeatureAnalysis")
                .getOrCreate();

       
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("your_data.csv");

        String[] featureColumns = data.columns();
        String[] inputCols = java.util.Arrays.stream(featureColumns)
                .filter(col -> !col.equals("Participant #") && !col.equals("PlayAttention Percentage"))
                .toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(data);

        ChiSqSelector selector = new ChiSqSelector()
                .setNumTopFeatures(10)
                .setFeaturesCol("features")
                .setLabelCol("PlayAttention Percentage")
                .setOutputCol("selectedFeatures");

        Dataset<Row> selectedData = selector.fit(assembledData).transform(assembledData);

        PCA pca = new PCA()
                .setK(2)
                .setInputCol("features")
                .setOutputCol("pcaFeatures");

        Dataset<Row> pcaResult = pca.fit(selectedData).transform(selectedData).select("pcaFeatures");

        pcaResult.show(false);

        pcaResult.write().csv("pca_output.csv");

        spark.stop();
    }
}
