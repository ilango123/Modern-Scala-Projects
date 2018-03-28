package com.packt.modern.chapter2.lr

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import scala.util.hashing.{MurmurHash3=>MH3}
import scala.math.{abs, max, min, log}
import collection.mutable.HashMap
import scala.io.{Source}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import java.io.FileWriter
import scala.util.control.Breaks._


//https://www.kaggle.com/rootua/apache-spark-scala-logistic-regression
class Logit {


}
