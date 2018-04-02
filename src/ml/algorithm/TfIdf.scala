package ml.algorithm

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TfIdf {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("tfidf")
      .master("local[*]")
      .getOrCreate()



    val df1 = spark.createDataFrame(Seq(
      (0,"a a a b b c")
      ,(0,"a b c")
      ,(1,"a c a a d")
    )).toDF("label","sentence")

    df1.show()

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    val df2 = tokenizer.transform(df1)

    df2.printSchema()

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("TF-features")

    hashingTF.transform(df2).show()

    val data = Seq("Tokenization is the process", "Refer to the Tokenizer").map(Tuple1(_))

    println(data)

    val inputDF = spark.createDataFrame(data).toDF("input")
    val tokenizer1 = new Tokenizer().setInputCol("input").setOutputCol("output")

    tokenizer1.transform(inputDF).head()
  }

}
