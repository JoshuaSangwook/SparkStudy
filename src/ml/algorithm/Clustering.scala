package ml.algorithm

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeansModel

object Clustering {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
        .builder()
        .appName("ClusteringTest")
        .master("local[*]")
        .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    val d1 = spark.read.option("header","true")
          .option("inferSchema",true)
        .csv("./data/wifidata.csv")


    d1.printSchema()

    val d2 = d1.toDF("number","name", "SI", "GOO", "DONG", "x", "y", "b_code", "h_code", "utmk_x", "utmk_y", "wtm_x", "wtm_y")

    d2.describe().show()

    val d3 = d2.select('GOO.as("loc"), 'x, 'y)

    val indexer = new StringIndexer().setInputCol("loc").setOutputCol("loccode")

    indexer.fit(d3).transform(d3).show()

    val d4 = indexer.fit(d3).transform(d3)
    val assembler = new VectorAssembler().setInputCols(Array("loccode","x","y")).setOutputCol("features")

    assembler.transform(d4).show()

    val kmeans = new KMeans().setK(5).setSeed(1L).setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(indexer,assembler,kmeans))

    val model = pipeline.fit(d3)

    val d5 = model.transform(d3)

    d5.groupBy("prediction").agg(collect_set("loc")).show()

    val WSSSE = model.stages(2).asInstanceOf[KMeansModel].computeCost(d5)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster Centers: ")
    model.stages(2).asInstanceOf[KMeansModel].clusterCenters.foreach(println)


  }

}
