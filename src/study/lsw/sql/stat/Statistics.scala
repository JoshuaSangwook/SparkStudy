package study.lsw.sql.stat

//import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._

case class MatchData(
                      id_1: Int,
                      id_2: Int,
                      cmp_fname_c1: Option[Double],
                      cmp_fname_c2: Option[Double],
                      cmp_lname_c1: Option[Double],
                      cmp_lname_c2: Option[Double],
                      cmp_sex: Option[Int],
                      cmp_bd: Option[Int],
                      cmp_bm: Option[Int],
                      cmp_by: Option[Int],
                      cmp_plz: Option[Int],
                      is_match: Boolean
                    )

object Statistics {
  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder
      .appName("Intro")
      .master("local[*]")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate

    import spark.implicits._

    val preview = spark.read
      .option("header", "true")
      .option("nullValue", "?")
      .option("inferSchema", "true")
      .csv("./data/block_1.csv")

    preview.printSchema()
    //preview.show(100)
    println(preview.count()) // 574913
    val summary = preview.describe()
    summary.show()
    summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()

    println(preview.where("is_match = true").count()) // 2093
    println(preview.where("is_match = false").count()) //572820

    val matches = preview.where("is_match = true")
    val misses = preview.filter($"is_match" === false)
    val matchSummary = matches.describe()
    val missSummary = misses.describe()

    val schema = matchSummary.schema

    longForm(matchSummary).show()
    pivotSummary(matchSummary).show()

    val matchData = preview.as[MatchData]
    matchData.printSchema()

    val scored = matchData.map { md =>
      (scoreMatchData(md), md.is_match)
    }.toDF("score", "is_match")


  }

  case class Score(value: Double) {
    def +(oi: Option[Int]) = {
      Score(value + oi.getOrElse(0))
    }
  }

  def scoreMatchData(md: MatchData): Double = {
    (Score(md.cmp_lname_c1.getOrElse(0.0)) + md.cmp_plz +
      md.cmp_by + md.cmp_bd + md.cmp_bm).value
  }

    def longForm(desc: DataFrame): DataFrame = {
      import desc.sparkSession.implicits._ // For toDF RDD -> DataFrame conversion
      val schema = desc.schema
      desc.flatMap(row => {
        val metric = row.getString(0)
        (1 until row.size).map(i => (metric, schema(i).name, row.getString(i).toDouble))
      })
        .toDF("metric", "field", "value")
    }

    def pivotSummary(desc: DataFrame): DataFrame = {
      val lf = longForm(desc)
      lf.groupBy("field").
        pivot("metric", Seq("count", "mean", "stddev", "min", "max")).
        agg(first("value"))
    }

}
