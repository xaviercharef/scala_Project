package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   // a) Charger un csv dans dataframe
   val df_trainer: DataFrame = spark
     .read
       .parquet("/home/charef/Documents/tp_scala_spark/prepared_trainingset").sample(withReplacement=false, 0.1)


    /** TF-IDF **/
    // a - tokenizer

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // b - stopWordRemover

    val stopWordSet =new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol).setOutputCol("filtered_token")

    // c - CountVectorizer

    val countVectorizer = new CountVectorizer()
      .setInputCol("filtered_token")
      .setOutputCol("word_count")


    // d - Trouvez la partie IDF

    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")


    // e - Convertir la variable catégorielle “country2” en données numérique

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("countryIndex")

    val encoder_country = new OneHotEncoder()
      .setInputCol("countryIndex")
      .setOutputCol("country_indexed")

    // f - Convertir la variable catégorielle “currency2” en données numérique

    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currencyIndex")

    val encoder_currency = new OneHotEncoder()
      .setInputCol("currencyIndex")
      .setOutputCol("currency_indexed")

    /** VECTOR ASSEMBLER **/

    // g - Assembler les features "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed",
    // "currency_indexed" dans une seule colonne “features”

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed",
        "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/

    // h - modèle de classification, il s’agit d’une régression logistique

    val logisticRegression = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/
   // i - créer le pipeline en assemblant les 8 stages définis précédemment, dans le​ ​ bon​ ​ ordre
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordSet, countVectorizer, idf, indexer_country, encoder_country,
        indexer_currency, encoder_currency, assembler, logisticRegression))

    pipeline.fit(df_trainer)
    df_trainer.show(6)

    /** TRAINING AND GRID-SEARCH **/

    // j - Créer un dataFrame nommé “training” et un autre nommé “test”
    val Array(df_training, df_test) = df_trainer.randomSplit(Array(0.9, 0.1))

    // k - Préparer la grid-search pour satisfaire les conditions explicitées ci-dessus puis
    // lancer​ ​ la​ ​ grid-search​ ​ sur​ ​ le​ ​ dataset​ ​ “training”​ ​ préparé​ ​ précédemment
    val paramGrid_countVectorizer = new ParamGridBuilder()
      .addGrid(countVectorizer.minDF, Array(55.toDouble, 95.toDouble, 20.toDouble))
      .addGrid(logisticRegression.regParam,Array(10e-8.toDouble,10e-6.toDouble,10e-4.toDouble,10e-2.toDouble))
      .build()




    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val validationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid_countVectorizer)


    val cvModel = validationSplit.fit(df_training)


    // l - Appliquer le meilleur modèle trouvé avec la grid-search aux données test
    val df_WithPredictions = cvModel.transform(df_test)

    val f1_Score = evaluator.evaluate(df_WithPredictions)
    println("F1 score on test data: " + f1_Score)


    // m -
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    df_WithPredictions.write
      .mode(SaveMode.Overwrite)
      .parquet("/home/charef/Documents/tp_scala_spark/trained_test")


  }
}
