package com.packt.modern.chapter2.lr

import com.packt.modern.chapter2.WisconsinWrapper
import com.packt.modern.chapter2.rf.BreastCancerRfPipeline$.bcwFeatures_IndexedLabel
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

object BreastCancerLrPipeline extends WisconsinWrapper {

  def main(args: Array[String]): Unit = {

    import org.apache.spark.ml.feature.StringIndexer

    //def setInputCols(value: Array[String]): VectorAssembler.this.type
    //def setOutputCol(value: String): VectorAssembler.this.type

    val indexer = new StringIndexer().setInputCol(bcwFeatures_IndexedLabel._2).setOutputCol(bcwFeatures_IndexedLabel._3)


  }

}
