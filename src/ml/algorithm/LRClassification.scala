package ml.algorithm

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object LRClassification {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Pipeline sample")
      .master("local[*]")
      .getOrCreate()

    // 훈련용 데이터 (키, 몸무게, 나이, 성별)
    val training = spark.createDataFrame(Seq(
      (161.0, 69.87,  5 , 1.0),
      (176.78, 74.35, 34, 1.0),
      (159.23, 58.32, 29, 0.0))).toDF("height", "weight", "age", "gender")

    training.printSchema()

    // 테스트용 데이터
    val test = spark.createDataFrame(Seq(
      (169.4, 75.3, 42),
      (185.1, 85.0, 37),
      (161.6, 61.2, 28))).toDF("height", "weight", "age")


    val assembler = new VectorAssembler()
      .setInputCols(Array("height", "weight", "age"))
      .setOutputCol("features")

    val assembled_training = assembler.transform(training)

    assembled_training.show()

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setLabelCol("gender")

    val model = lr.fit(assembled_training)

    model.transform(assembled_training).show()

    val pipeline = new Pipeline().setStages(Array(assembler,lr))

    val pipelineModel = pipeline.fit(training)

    pipelineModel.transform(training).show()

    val path1 = "./result/a"
    val path2 = "./result/b"

    model.write.overwrite().save(path1)
    pipelineModel.write.overwrite().save(path2)

    // 저장된 모델 불러오기
    val loadedModel = LogisticRegressionModel.load(path1)
    val loadedPipelineModel = PipelineModel.load(path2)

    val assembled_test = assembler.transform(test)

    loadedModel.transform(assembled_test).show()
    loadedPipelineModel.transform(test).show()

  }

}
