import org.apache.spark.sql.SparkSession

object WordCount {
  def main(args: Array[String]): Unit = {
    // Create SparkSession
    val spark = SparkSession.builder
      .appName("Word Count")
      .master("local[*]") // Run locally with all cores
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    val sc = spark.sparkContext

    // Read the input file (local or HDFS)
    val textFile = sc.textFile("sample.txt")

    // Word count logic
    val counts = textFile
      .flatMap(line => line.split(" "))   // Split into words
      .map(word => (word, 1))             // Map each word to (word, 1)
      .reduceByKey(_ + _)                 // Reduce by key to get counts

    // Show results
    counts.collect().foreach(println)

    // Stop the SparkSession
    spark.stop()
  }
}


'''sample.txt
hello world
hello spark
spark is fast
'''

'''bash
scalac MyApp.scala
scala MyApp
'''
'''Apache spark
spark-submit --class WordCount WordCount.scala
'''
