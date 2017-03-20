package com.coviam.sentimentAnalysis.engine

import org.apache.predictionio.controller.{IPersistentModel, IPersistentModelLoader, P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, SQLContext}



class Algorithm(val ap:AlgorithmParams) extends P2LAlgorithm[PreparedData, NBModel, Query, PredictedResult]{

  override def train(sc: SparkContext, pd: PreparedData): NBModel = {
    val nb = NaiveBayes.train(pd.labeledpoints,lambda = ap.lambda,modelType = "multinomial")
    print("trained")
    NBModel(nb,sc)
  }


  override def predict(model: NBModel, query: Query): PredictedResult = {


    print("prediction called")

    val sql = SQLContext.getOrCreate(model.sc)
    val phrase = sql.createDataFrame(Seq(query)).toDF("phrase")
    val obj = new DataPreparator(PreparatorParams(1))
    val tf = obj.processPhrase(phrase)
    tf.show(10)
    val labeledpoints = tf.map(row => row.getAs[Vector]("rowFeatures"))

    print("label points --",labeledpoints)

    val predictedResult = model.nb.predict(labeledpoints)
    val result = predictedResult.first()
    val prob = model.nb.predictProbabilities(labeledpoints)
    val score = prob.first().toArray
    var weight:Double = 0.0
    if(result == 1.0)
      weight = score.last
    else
      weight = score.head

    PredictedResult(result,weight)
  }

  override def batchPredict(m: NBModel, qs: RDD[(Long, Query)]): RDD[(Long, PredictedResult)] = {
    qs.sparkContext.parallelize(
      qs.collect().map{
        case (index, query) => (index, predict(m, query))
      }
    )
  }
}


case class AlgorithmParams(val lambda:Double) extends Params

case class NBModel( nb: NaiveBayesModel,
                     sc: SparkContext
                   ) extends IPersistentModel[AlgorithmParams] with Serializable{

  def save(id: String, params: AlgorithmParams, sc: SparkContext): Boolean = {
    nb.save(sc, s"/tmp/${id}/nbModel")
    true
  }

}

object NBModel extends IPersistentModelLoader[AlgorithmParams, NBModel]{
  def apply(id: String, params: AlgorithmParams, sc: Option[SparkContext]) = {
    new NBModel(
      NaiveBayesModel.load(sc.get,s"/tmp/${id}/nbModel"),
      sc.get
    )
  }
}


//    extra code
//    val sc_new = SparkContext.getOrCreate()
//    val sqlContext = SQLContext.getOrCreate(sc_new)
//    val phraseDataframe = sqlContext.createDataFrame(Seq(query)).toDF("phrase")
//    phraseDataframe.show()
//
//    val dpObj = new DataPreparator
//    val tf = dpObj.processPhrase(phraseDataframe)
//    tf.show()
//    val labeledpoints = tf.map(row => row.getAs[Vector]("rowFeatures"))
//    print("label points --",labeledpoints)
//    val predictedResult = model.nb.predict(labeledpoints)
//    val result = predictedResult.first()
//
//    val prob = model.nb.predictProbabilities(labeledpoints)
//    val score = prob.first().toArray
//
//    var weight:Double = 0.0
//    if(result == 1.0)
//      {weight = score.last}
//    else
//      {weight = score.head}
//    print("score --",score)
//    PredictedResult(result,weight)
