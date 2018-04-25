/*
 * The MIT License
 *
 * Copyright 2017 Matías Roodschild <mroodschild@gmail.com>.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.gitia.clase2017.rnn.tp2;


import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.Feedforward;
import org.gitia.froog.layer.Layer;
import org.gitia.froog.lossfunction.LossFunction;
import org.gitia.froog.statistics.Compite;
import org.gitia.froog.statistics.ConfusionMatrix;
import org.gitia.froog.trainingalgorithm.Backpropagation;
import org.gitia.froog.trainingalgorithm.SGD;
import org.gitia.froog.transferfunction.TransferFunction;
import org.gitia.jdataanalysis.CSV;
import org.gitia.jdataanalysis.data.stats.FilterConstantColumns;
import org.gitia.jdataanalysis.data.stats.STD;

/**
 *
 * @author Matías Roodschild <mroodschild@gmail.com>
 */
public class MnistExample {

    public static void main(String[] args) {

        SimpleMatrix input = CSV.open("src/main/resources/mnist/train_in.csv");
        SimpleMatrix output = CSV.open("src/main/resources/mnist/train_out.csv");
        SimpleMatrix in_test = CSV.open("src/main/resources/mnist/test_in.csv");
        SimpleMatrix out_test = CSV.open("src/main/resources/mnist/test_out.csv");

        FilterConstantColumns filter = new FilterConstantColumns();
        filter.fit(input);//ajustamos el filtro
        System.out.println("Dimensiones iniciales");
        input.printDimensions();
        input = filter.eval(input);
        in_test = filter.eval(in_test);
        System.out.println("Dimensiones finales");
        input.printDimensions();

        STD std = new STD();
        std.fit(input);
        input = std.eval(input);
        in_test = std.eval(in_test);

        int inputSize = input.numCols();
        int outputSize = output.numCols();
        
        //==================== Preparamos la RNA =======================
        Random rand = new Random(1);
        Feedforward net = new Feedforward();
        net.addLayer(new Layer(inputSize, 300, TransferFunction.TANSIG, rand));
        net.addLayer(new Layer(300, 150, TransferFunction.TANSIG, rand));
        net.addLayer(new Layer(150, outputSize, TransferFunction.SOFTMAX, rand));

        //=================  configuraciones del ensayo ========================
        // Preparamos el algoritmo de entrenamiento
        SGD sgd = new SGD();
        sgd.setEpoch(1);//10
        sgd.setBatchSize(1000);//10
        sgd.setMomentum(0.9);
        sgd.setLearningRate(0.01);
        sgd.setInputTest(in_test.transpose());
        sgd.setOutputTest(out_test.transpose());
        sgd.setTestFrecuency(2000);
        sgd.setClassification(true);
        sgd.setLossFunction(LossFunction.CROSSENTROPY);
        
        sgd.train(net, input.transpose(), output.transpose());
        
        SimpleMatrix out1 = Compite.eval(net.output(input.transpose()).transpose());
        System.out.println("\nMatriz de Confusion 1 - Datos de entrenamiento");
        ConfusionMatrix cmTrain = new ConfusionMatrix();
        out1.printDimensions();
        output.printDimensions();
        cmTrain.eval(out1, output.transpose());
        cmTrain.printStats();

        SimpleMatrix out2 = Compite.eval(net.output(in_test.transpose()).transpose());
        System.out.println("\nMatriz de Confusion 2 - Datos de Testeo");
        ConfusionMatrix cmTest = new ConfusionMatrix();
        cmTest.eval(out2, out_test.transpose());
        cmTest.printStats();
    }
}