package com.coviam.sentimentAnalysis.engine

import org.apache.predictionio.controller.{PPreparator, Params}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Tokenizer, _}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

//PreparatorParams to check accuracy with unigram/bigram as feature

class DataPreparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData]{

  override def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val obs = trainingData.phraseAndSentiment
    val sqlContext = SQLContext.getOrCreate(sc)
    val phraseDataframe = sqlContext.createDataFrame(obs).toDF("phrase", "sentiment")
    val tf = processPhrase(phraseDataframe)
    val labeledpoints = tf.map(row => new LabeledPoint(row.getAs[Double]("sentiment"), row.getAs[Vector]("rowFeatures")))
    PreparedData(labeledpoints)
  }

  def processPhrase(phraseDataframe:DataFrame): DataFrame ={

    val tokenizer = new Tokenizer_new().setInputCol("phrase").setOutputCol("unigram")
    val unigram = tokenizer.transform(phraseDataframe)

    //use below ngram feature to test accuracy with unigram or bigram as feature
    //val ngram = new NGram().setN(pp.ngram).setInputCol("unigram").setOutputCol("ngrams")

    // user below ngram feature to test aacuracy with unigram+bigram as feature
    val ngram = new Ngram_new().setInputCol("unigram").setOutputCol("ngrams")
    val ngramDataFrame = ngram.transform(unigram)

    val remover = new StopWordsRemover().setInputCol("ngrams").setOutputCol("filtered")
    val stopRemoveDF = remover.transform(ngramDataFrame)

    var htf = new HashingTF().setInputCol("filtered").setOutputCol("rowFeatures")
    val tf = htf.transform(stopRemoveDF)

    tf
  }
}

case class PreparedData(var labeledpoints:RDD[LabeledPoint]) extends Serializable{
}

class Tokenizer_new extends Tokenizer(){

  override def createTransformFunc: (String) => Seq[String] = { str =>
    val NEGATION = "never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|ain|aint|n't"
    val unigram = str.toLowerCase().replaceAll("\n", "")
                  .replaceAll("rt\\s+", "")
                  .replaceAll("\\s+@\\w+", "")
                  .replaceAll("@\\w+", "")
                  .replaceAll("\\s+#\\w+", "")
                  .replaceAll("#\\w+", "")
                  .replaceAll("(?:https?|http?)://[\\w/%.-]+", "")
                  .replaceAll("(?:https?|http?)://[\\w/%.-]+\\s+", "")
                  .replaceAll("(?:https?|http?)//[\\w/%.-]+\\s+", "")
                  .replaceAll("(?:https?|http?)//[\\w/%.-]+", "")
                  .split("\\W+")
                  .filter(_.matches("^[a-zA-Z]+$")).filter(x => x.length>2)

    (1 until unigram.length).foreach(i=>
      if(unigram(i-1).matches(NEGATION)){
        for(x <- i to unigram.length-1)
        {
          unigram(x) += "_NEG"
        }
      }
      else{
        unigram(i-1)
      }
    )
    unigram.toSeq
  }
}

class Ngram_new extends NGram(){

  override def createTransformFunc: (Seq[String]) => Seq[String] = { r =>
    val bigram = (1 until(r.length)).map(
      i => r(i-1) ++" "++ r(i)
    )
    val ngram = r ++ bigram
    ngram
  }

}


case class PreparatorParams(var ngram:Int)extends Params

