package ml.algorithm

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Classification {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("ClassficationSample")
      .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
      .getOrCreate()

    import org.apache.spark.sql.functions._
    import spark.implicits._

     val df1 = spark.read.option("header", "true")
       .option("sep", ",").option("inferSchema", true)
       .option("mode", "DROPMALFORMED")
       .csv("./data/seouldata.csv")

    val d2 = df1.toDF("year", "month", "road", "avr_traffic_month", "avr_velo_month", "mon", "tue", "wed", "thu", "fri", "sat", "sun")

    d2.show()
    d2.describe().show()
    println(d2.where("mon is null").count())
    d2.where("mon is null").show()

    val d3 = d2.where("mon is not null")
    val avgRoad = d3.groupBy("road").agg(round(avg("avr_velo_month"), 1).as("avr_velo_total"))

    val d5 = d3.join(avgRoad,Seq("road"))

    // Label(혼잡:1.0, 원활:0.0)  136/132
    spark.udf.register("label", ((avr_month: Double, avr_total: Double) => if ((avr_month - avr_total) >= 0) 1.0 else 0.0))

    val d6 = d5.withColumn("label", callUDF("label", $"avr_velo_month", $"avr_velo_total"))
    d6.select("road", "avr_velo_month", "avr_velo_total", "label").show(5, false)
    d6.groupBy("label").count().show(false)

    val Array(train,test) = d6.randomSplit(Array(0.7,0.3))

    println(train.count())  //178
    println(test.count())   //70


    val indexer = new StringIndexer().setInputCol("road").setOutputCol("roadcode")

    val assembler = new VectorAssembler()
                          .setInputCols(Array("roadcode","mon","tue", "wed", "thu", "fri", "sat", "sun"))
                          .setOutputCol("features")

    val dt = new DecisionTreeClassifier()
                      .setLabelCol("label")
                      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(indexer,assembler,dt))

    val model = pipeline.fit(train)

    val predict = model.transform(test)

    predict.select("label", "probability", "prediction").show()

    val evaluator = new BinaryClassificationEvaluator()
                            .setLabelCol("label")
                            .setMetricName("areaUnderROC")

    println(evaluator.evaluate(predict))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)


  }

}
