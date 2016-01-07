package test

import weka.core.FastVector
import weka.core.Attribute
import weka.core.Instances
import weka.core.Instance
import weka.associations.tertius.IndividualInstance
import weka.core.SparseInstance
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.classifiers.functions.LinearRegression
import weka.classifiers.trees.RandomTree
import weka.classifiers.bayes.BayesNet
import weka.classifiers.functions.LeastMedSq
import weka.classifiers.functions.GaussianProcesses
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.pmml.consumer.NeuralNetwork.Neuron
import weka.classifiers.pmml.consumer.NeuralNetwork.Neuron
import weka.classifiers.pmml.consumer.NeuralNetwork
import weka.classifiers.pmml.consumer.GeneralRegression

object Main {

  def main(args: Array[String]) {

    var attributeA = new Attribute("a");
    var attributeB = new Attribute("b");
    var attributeResult = new Attribute("result");

    var fvWekaAttributes = new FastVector(4);
    fvWekaAttributes.addElement(attributeA);
    fvWekaAttributes.addElement(attributeB);
    fvWekaAttributes.addElement(attributeResult);

    // Create an empty training set
    var isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);
    isTrainingSet.setClassIndex(fvWekaAttributes.size() - 1);

    var iExample1 = new SparseInstance(3)
    iExample1.setValue(attributeA, 0);
    iExample1.setValue(attributeB, 1);
    iExample1.setValue(attributeResult, 0);

    var iExample2 = new SparseInstance(3)
    iExample2.setValue(attributeA, 1);
    iExample2.setValue(attributeB, 2);
    iExample2.setValue(attributeResult, 2);

    var iExample3 = new SparseInstance(3)
    iExample3.setValue(attributeA, 2);
    iExample3.setValue(attributeB, 3);
    iExample3.setValue(attributeResult, 6);

    var iExample4 = new SparseInstance(3)
    iExample4.setValue(attributeA, 3);
    iExample4.setValue(attributeB, 4);
    iExample4.setValue(attributeResult, 12);

    var iExample5 = new SparseInstance(3)
    iExample5.setValue(attributeA, 4);
    iExample5.setValue(attributeB, 5);
    iExample5.setValue(attributeResult, 20);

    var iExample6 = new SparseInstance(3)
    iExample6.setValue(attributeA, 5);
    iExample6.setValue(attributeB, 6);
    iExample6.setValue(attributeResult, 30);

    var iExample7 = new SparseInstance(3)
    iExample7.setValue(attributeA, 6);
    iExample7.setValue(attributeB, 7);
    iExample7.setValue(attributeResult, 42);

    var iExample8 = new SparseInstance(3)
    iExample8.setValue(attributeA, 7);
    iExample8.setValue(attributeB, 8);
    iExample8.setValue(attributeResult, 56);

    var iExample9 = new SparseInstance(3)
    iExample9.setValue(attributeA, 8);
    iExample9.setValue(attributeB, 9);
    iExample9.setValue(attributeResult, 72);

    isTrainingSet.add(iExample1);
    isTrainingSet.add(iExample2);
    isTrainingSet.add(iExample3);
    isTrainingSet.add(iExample4);
    isTrainingSet.add(iExample5);
    isTrainingSet.add(iExample6);
    isTrainingSet.add(iExample7);
    isTrainingSet.add(iExample8);
    isTrainingSet.add(iExample9);

    var iExampleX = new SparseInstance(3)
    iExampleX.setValue(attributeA, 4);
    iExampleX.setValue(attributeB, 2);

    //Instance of a Neural Network
    var mlp = new MultilayerPerceptron();
    //Setting Parameters
    mlp.setLearningRate(0.1);
    mlp.setMomentum(0.2);
    mlp.setTrainingTime(1000000);
    mlp.setHiddenLayers("5");
    mlp.buildClassifier(isTrainingSet);

    println(mlp.classifyInstance(iExampleX))

  }

}