package com.packt.modern.chapter1

import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import breeze.linalg.support.LiteralRow._
import scala.reflect.io.File
import java.io.PrintWriter
import java.io.FileWriter

object PrepareData extends App{
//Just because the object extends the App trait, all the code inside it will automatically run, as if it were inside a main method
  

  //This works for iris.data by replacing the values in the 5th column with 
  def transformLabel(aFlower:Array[String]):Array[String] = aFlower(4) match {
    case "Iris-setosa" => aFlower.updated(4, "0")
    case "Iris-versicolor" => aFlower.updated(4, "1")
    case "Iris-virginica" => aFlower.updated(4, "2")
    case _ => Array.empty[String]
  } 
  
  val source = Source.fromFile("iris.data"); //data with no header, for e.g. 5.1,3.5,1.4,0.2,Iris-setosa
  
    //this value contains 
    val dataPreprocessed = {
	  try {
		val lines = source.getLines().flatMap(line => transformLabel(line.split(",")))
		lines.toList
	  } catch {
		// re-throw exception, but make sure source is closed
		case
		  t: Throwable => {
		  println("error during parsing of file")
		  throw t
		}
	  } finally source.close()
	  
  }
   
 
    println("dataPreprocessed is: " + dataPreprocessed)
  
  
    /** Create a Matrix out of that array.**/
  
	 val denseMatrix = new DenseMatrix(150,5,dataPreprocessed.toArray)
	
     println("------------------------------------------")
     println("denseMatrix is: " + denseMatrix)

}


