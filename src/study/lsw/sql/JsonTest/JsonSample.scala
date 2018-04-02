package study.lsw.sql.JsonTest

import play.api.libs.json._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.util.QueryExecutionListener

object JsonSample  {

  case class Inquiry(year: String, month: String, day: String, path: String, columns: String*)

  def main(args: Array[String]): Unit = {



  implicit val modelFormat = Json.format[Inquiry]

    val js = """{"year":"2018","month":"02","day":"03","path":"/MART/APP/mart_app_vst_ctnt","columns":["MBR_ID","VSTR_ID"]}"""

  val inq = Json.fromJson[Inquiry](Json.parse(js)).get

  val b1d_yyyy = inq.year
  val b1d_mm = inq.month
  val b1d_dd = inq.day
  val path = inq.path
  val columns = inq.columns.toList.map(col(_))


    val spark =
      SparkSession.builder
        .config("key", "love")
        .appName("json")
        .enableHiveSupport()
        .getOrCreate()



    println(b1d_yyyy)

    //테스트를 위해 의미없는 hive partition 생성
    spark.sql(
      s"""
        ALTER TABLE user_bi_ocb.wk_mart_app_bd_thm_ctnt
        ADD IF NOT EXISTS PARTITION (base_dt='$b1d_yyyy$b1d_mm$b1d_dd')
        LOCATION '/data_bis/ocb/MART/APP/WK/wk_mart_app_bd_thm_ctnt/$b1d_yyyy/$b1d_mm/$b1d_dd'
      """)

    spark.read
      .orc(s"$path/$b1d_yyyy/$b1d_mm/$b1d_dd/poc_fg_cd=01")
      .select(columns:_*)
      .withColumn("ld_dttm", from_unixtime(unix_timestamp(), "yyyyMMddHHmmss"))
      .withColumn("mart_upd_dttm", from_unixtime(unix_timestamp(), "yyyyMMddHHmmss"))
      .write
      .mode("overwrite")
      .orc(s"/temp/spark-sample")


  }
}
