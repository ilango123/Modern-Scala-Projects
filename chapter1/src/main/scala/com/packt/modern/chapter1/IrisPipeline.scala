package com.packt.modern.chapter1

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame

import scala.reflect.runtime.universe
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.tuning.TrainValidationSplit

trait Iris {

	//The entry point to programming Spark with the Dataset and DataFrame API.
	//This is the SparkSession
	
	lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("iris-pipeline")
      .getOrCreate()
     }
	 
	 
  
    def irisFeatureColumn = "iris-features"
    def irisTypeColumn = "iris-type"
	  /**
	   * Load iris data. This data is a CSV with no header.
	   * 
	   * The description of this dataset is found in iris.names. A detailed description of this
	   * is also to be found in the Project Overview section.
	   * 
	   *  filePath will be "C:/Users/Ilango/Documents/Packt-Book-Writing-Project/Trial_And_Error_Projects/chapter1/iris.data"
	   *  or simply "iris.data", because the Spark Shell is started from the chapter1 folder
	   * @return a Dataframe with two columns. `irisFeatureColumn` contains the feature `Vector`s and `irisTypeColumn` contains the `String` iris types
	   */
  def loadIris(filePath: String): DataFrame = {
    val irisData = spark.sparkContext.textFile(filePath).flatMap { text =>
      text.split("\n").toList.map(_.split(",")).collect {
        case Array(sepalLength, sepalWidth, petalLength, petalWidth, irisType) =>
          (Vectors.dense(sepalLength.toDouble, sepalWidth.toDouble, petalLength.toDouble, petalWidth.toDouble), irisType)
      }
    }
	 
	  println("irisData is:" + irisData)
   
	spark.createDataFrame(irisData).toDF(irisFeatureColumn, irisTypeColumn)
  }//end of function loadIris
  
  
}



/**
 * SpeciesClassifier is where we create a Classififier and a machine learning pipeline to apply on it
 */
object SpeciesClassifier extends Iris {
  
  /**
   * Only expects a single arg
   * arg(0) should have the path to the iris data
   */
   def main(args: Array[String]): Unit = {
    
    //val conf = new SparkConf(true).setAppName("iris-pipeline")
	//val sparkContext = new SparkContext(sparkConf)
	import spark.implicits._
    
    //load the iris.data file
	//val irisDataFrame = loadIris(args(0))
	val irisDataFrame = loadIris("iris.data")
	println("iris data frame is: " + irisDataFrame)
    val (trainingData, testData) = {
      // Experiment with adjusting the size of the training set vs the test set
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }
    
    /*  Build the Pipeline: Build a model, train it with a RandomForestClassifier
	 *	A multiclass classifier using a collection of decision trees. This classifier will create
     *  a model for predicting the "class" (i.e. iris type) of a flower based on its measurements
     *  
     *  StringIndexer:
     *  The iris types are all Strings needing to be indexed (i.e. turned into unique doubles)
     *  In order to work with a classifier. e.g. "Iris-setosa" might become 1.0 or a .1
     *          
     *  The Pipeline looks like this: Indexer -> Classifier
     */
    val indexer = new StringIndexer().setInputCol(irisTypeColumn).setOutputCol("label")
    
	val classifier = new RandomForestClassifier().setFeaturesCol(irisFeatureColumn)
    val pipeline = new Pipeline().setStages(Array(indexer, classifier))
      
    /*
     *
	 * To arrive at the best possible parameter to tune the accuracy of the classifier,we
	 * build a parameter grid to test each combination for its effectivenessm, rather than try 
	 * one parameter after another
     */
    val grid = new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(2, 5, 10))
      .addGrid(classifier.numTrees, Array(10, 20, 40))
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .build()
    
    //we break up the data into a training set that is 80% of the data and the remaining 20% to validate
	val trainValidation = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(grid)
      .setTrainRatio(0.8)
    
    // Create our model with the training data 
    trainingData.cache()
	//Lets train our random forest classsifer to take the training features and 
    val randomForestModel = trainValidation.fit(trainingData)
    
    // Use the model with our test-set of data
    testData.cache()
    val testResults = randomForestModel.transform(testData)
	println("testResults is: " + testResults)
    
    /*
     * Our model added 2 columns, a 'probability' vector and a 'prediction' vector.
	  * The probability column shows the odds the given flower is iris species iris_i for all i
     *  Fore example,  [0.0, 0.4, 0.6] translates to 0% chance it is iris_0.0, 40% chance it
     *  is iris_1.0, 60% chance it is iris_2.0
     * - 'prediction' the label for the iris type that our model concludes this row should be classified
     *    as. e.g. 2.0
     *    
     * Then we compare the predicted label in `prediction` to the actual label in `label` to see just how well our classifier did
     * 
     */
	
     val predictionAndLabels = testResults.select("prediction", "label") .map { 
											case Row(prediction: Double, label: Double) => 
							(prediction, label)
        }
	 
	 
     val metrics = new MulticlassMetrics(predictionAndLabels.rdd)
     println(s"Precision is ${metrics.precision}")
     println(s"Recall ${metrics.recall}")
     println(s"Feature importance Score ${metrics.fMeasure}")
    
	 spark.sparkContext.stop
  }
  
}