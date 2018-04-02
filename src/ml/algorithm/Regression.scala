package ml.algorithm

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object Regression {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("regression")
      .master("local[*]")
      .getOrCreate()

    val df1 = spark.read
        .option("sep","\t")
        .option("header",false)
        .csv("./data/report.txt")

    df1.show()
    val df2 = df1.where(df1("_c0") =!= "기간")

    spark.udf.register("toDouble", (v:String) => {
      v.replaceAll("[^0-9.]","").toDouble
    })

    import spark.implicits._
    df2.printSchema()

    val df3 = df2.select(
      df2("_c0").as("year")
      ,callUDF("toDouble",df2("_c2")).as("height")
      ,df2("_c4").cast("Double").as("weight")
    )
        .withColumn("grade", lit("elementary"))
        .withColumn("gender",lit("man"))

    df3.printSchema()
    // 초등학교 여 키, 몸무게
    val df4 = df2.select('_c0.as("year"),
      callUDF("toDouble", '_c3).as("height"),
      callUDF("toDouble", '_c5).as("weight"))
      .withColumn("grade", lit("elementary"))
      .withColumn("gender", lit("woman"))

    // 중학교 남 키, 몸무게
    val df5 = df2.select('_c0.as("year"),
      callUDF("toDouble", '_c6).as("height"),
      callUDF("toDouble", '_c8).as("weight"))
      .withColumn("grade", lit("middle"))
      .withColumn("gender", lit("man"))

    // 중학교 여 키, 몸무게
    val df6 = df2.select('_c0.as("year"),
      callUDF("toDouble", '_c7).as("height"),
      callUDF("toDouble", '_c9).as("weight"))
      .withColumn("grade", lit("middle"))
      .withColumn("gender", lit("woman"))

    // 고등학교 남 키, 몸무게
    val df7 = df2.select('_c0.as("year"),
      callUDF("toDouble", '_c10).as("height"),
      callUDF("toDouble", '_c12).as("weight"))
      .withColumn("grade", lit("high"))
      .withColumn("gender", lit("man"))

    // 고등학교 여 키, 몸무게
    val df8 = df2.select('_c0.as("year"),
      callUDF("toDouble", '_c11).as("height"),
      callUDF("toDouble", '_c13).as("weight"))
      .withColumn("grade", lit("high"))
      .withColumn("gender", lit("woman"))

    val df9 = df3.union(df4).union(df5).union(df6).union(df7).union(df8)

    // 연도, 키, 몸무게, 학년, 성별
    df9.show(5, false)
    df9.printSchema()

    val genderIndexer = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("gendercode")

    val gradeIndexer = new StringIndexer()
      .setInputCol("grade")
      .setOutputCol("gradecode")

    genderIndexer.fit(df9).transform(df9).show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("height","gendercode","gradecode"))
      .setOutputCol("features")

    val lr = new LinearRegression()
      .setMaxIter(5)
      .setRegParam(0.3)
      .setLabelCol("weight")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
        .setStages(Array(genderIndexer,gradeIndexer,assembler,lr))

    pipeline.fit(df9).transform(df9).show()


    val Array(training,test) = df9.randomSplit(Array(0.7,0.3))
    println(training.count(),test.count())

    val model = pipeline.fit(training)

    model.transform(test)
      .select("weight","prediction")
      .show()

    //println("r2 : " + lr.fit(training).summary.r2)

  }
}
