package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.udf
object Preprocessor_training {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/
    //var dataFile_RDD = sc.textFile("/home/charef/Documents/train.csv")
    //val dataframe_train = dataFile_RDD.toDF()
    val dataframe_train = spark.read.format("csv").option("inferSchema","true").option("header", true).load("/home/charef/Documents/train.csv")
    // compte les lignes
    println(dataframe_train.count())
    // compte les colonnes
    print(dataframe_train.columns.length)
    // affiche le schema
    print(dataframe_train.printSchema())
    // cast les colonnes censées avoir des entiers
    val dataframe_with_good_type = dataframe_train
      .withColumn("currency",dataframe_train("currency").cast("Int"))
      .withColumn("backers_count",dataframe_train("backers_count").cast("Int"))
      .withColumn("final_status",dataframe_train("final_status").cast("Int"))
    // affiche le schema du dataframe
    print(dataframe_with_good_type.printSchema())
    dataframe_train.show()
    /** 2 - CLEANING **/

    dataframe_with_good_type.groupBy("final_status").count.orderBy($"count".desc).show
    // description statistique
    dataframe_with_good_type.select("goal","backers_count","")

    val dfNoFtur : DataFrame = dataframe_with_good_type
      .drop("backers_count","state_changed_at")

    def udf_country = udf {(country: String, currency: String) =>
      if (country == null)
        currency
      else
        country
    }
    /**
    def udf_currency = udf {(currency: String) =>
      if (currency != null && currency.length <= 2)
        currency
      else

    }**/

    //val data_without_series_quotes = dataframe_train.withColumn("replaced",regexp_replace(dataframe_train("name"),"\"{2,}"))
    //df2.select("replaced").coalesce(1).write.text(".../train.csv")
    //val df2= df.withColumn("replaced", regexp_replace($"value","\"{2,}"))
    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/
    /**val dfDurations: DataFrame = dflower.withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))
      .withColumn("days_campaign", datediff($"launched_at" - $"created_at")/3600.0,3)
      .filter($"hours_prepa" >= 0 && $"days_campaign" >= 0)
      .drop("created_at","deadline","launched_at")

    val dfText = dfDurations.withColumn("text", concat_ws("",$"name",$"desc", $"keywords"))

    val dffFiltered = dfText.filter($"final_status", isin(0,1))
    /** DATA WRITING**/
    dffFiltered.write.mode(SaveMode.Overwrite).parquet("/.../machin.csv")**/
  }


}
