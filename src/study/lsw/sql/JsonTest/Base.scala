package study.lsw.sql.JsonTest

import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory
import play.api.libs.json._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.util.QueryExecutionListener
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.QueryExecution


class Base  {

  case class Argument(
                         anything: Option[String] //= Some("")
                       , feature:  Option[String] //= Some("")
                       , year:     Option[String] //= Some("")
                       , month:    Option[String] //= Some("")
                       , day:      Option[String] //= Some("")
                       , path:     Option[String] //= Some("")
                       , columns:  Option[List[String]] //= Some(Nil)
                     )



    val js = """{"year":"2018","month":"02","day":"03","path":"/MART/APP/mart_app_vst_ctnt","columns":["MBR_ID","VSTR_ID"]}"""

    implicit val modelFormat = Json.using[Json.WithDefaultValues].format[Argument]
    protected val newArgs = Json.fromJson[Argument](Json.parse(js)).get





}
