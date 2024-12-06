import org.apache.spark.sql.{Row, SparkSession}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import org.apache.spark.ml.Pipeline

object SparkNlp {

  def main(args: Array[String]): Unit = {

    val spark = createSparkSession()
    val data = loadData(spark, "/Users/ahmad/IdeaProjects/SparkNLP/source/dataset.txt")

    val pipeline = buildPipeline()
    val pipelineModel = pipeline.fit(data)
    val result = pipelineModel.transform(data)

    result.printSchema()

    result.select("token.result", "pos.result", "ner.result").show(truncate = false)

    analyzeRelationships(result)

    spark.stop()
  }

  private def createSparkSession(): SparkSession = {
    SparkSession.builder()
      .appName("Spark NLP Advanced")
      .master("local[*]")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1")
      .getOrCreate()
  }

  spark.sparkContext.setLogLevel("Error")

  private def loadData(spark: SparkSession, filePath: String) = {
    spark.read.text(filePath).toDF("text")
  }

  private def buildPipeline(): Pipeline = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val wordEmbeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("embeddings")

    val posTagger = PerceptronModel.pretrained("pos_anc", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")

    val nerCrfModel = NerCrfModel.pretrained("ner_crf", "en")
      .setInputCols(Array("document", "token", "pos", "embeddings"))
      .setOutputCol("ner")

    new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      wordEmbeddings,
      posTagger,
      nerCrfModel
    ))
  }

  private def analyzeRelationships(result: org.apache.spark.sql.DataFrame): Unit = {
    val analyzeRelationships = result.rdd.map(row => {
      val posTags = row.getAs[Seq[Row]]("pos").map(_.getAs[String]("result"))  // Access pos column correctly
      val nerEntities = row.getAs[Seq[Row]]("ner").map(_.getAs[String]("result"))  // Access ner column correctly

      val entityPosMapping = nerEntities.zip(posTags).map {
        case (entity, posTag) =>
          val relationshipExplanation = analyzePosTag(entity, posTag)
          (entity, posTag, relationshipExplanation)
      }

      (row.getAs[String]("text"), entityPosMapping)
    }).collect()

    analyzeRelationships.foreach {
      case (text, mapping) =>
        println(s"\nText: $text")
        mapping.foreach {
          case (entity, posTag, explanation) =>
            println(s"  Entity: $entity, POS Tag: $posTag, Explanation: $explanation")
        }
    }
  }

  private def analyzePosTag(entity: String, posTag: String): String = {
    posTag match {
      case "NNP" => s"$entity seems to be a proper noun, likely a person, location, or organization."
      case "NN"  => s"$entity seems to be a common noun, likely an object or thing."
      case "VB"  => s"$entity seems to be a verb, potentially an action."
      case "JJ"  => s"$entity seems to be an adjective, describing a noun."
      case "RB"  => s"$entity seems to be an adverb, modifying a verb, adjective, or another adverb."
      case _     => s"Entity '$entity' with POS tag '$posTag' might represent something specific."
    }
  }
}
